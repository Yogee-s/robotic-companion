"""Adaptive barge-in detector.

Replaces the old fixed-threshold RMS check at `manager.py:216` with a
three-layer detector that distinguishes real user speech from:

* the robot's own TTS leaking back through the mic (envelope AEC-lite),
* ambient noise (adaptive noise floor),
* transients like door slams (sustained-speech requirement).

Interface
---------
    detector = BargeInDetector(vad)
    detector.note_tts_sample(pcm, sr)       # while speaking
    if detector.should_interrupt(chunk):
        cancel_turn()

Only `should_interrupt()` is called from the main audio loop; all other
methods are thread-safe and lock-free.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

log = logging.getLogger(__name__)


class BargeInDetector:
    def __init__(
        self,
        vad,
        *,
        noise_floor_alpha: float = 0.02,   # EMA smoothing on non-speech floor
        snr_margin_db: float = 10.0,       # must exceed floor by this much
        min_sustained_ms: int = 120,       # require this much voiced speech
        tts_subtract_gain: float = 0.6,    # how much of TTS envelope to subtract
        tts_envelope_window_s: float = 1.0,
    ) -> None:
        self._vad = vad
        self._noise_floor = 1e-4           # seed above zero
        self._noise_alpha = float(noise_floor_alpha)
        self._snr_margin_db = float(snr_margin_db)
        self._min_sustained_s = min_sustained_ms / 1000.0
        self._tts_subtract_gain = float(tts_subtract_gain)
        self._tts_env_window_s = float(tts_envelope_window_s)
        self._tts_envelope: Deque[tuple[float, float]] = deque(maxlen=128)
        self._speech_streak_start: Optional[float] = None

    # ── feedback from the TTS side ──────────────────────────────────────
    def note_tts_sample(self, pcm_int16: np.ndarray, sample_rate: int) -> None:
        """Called after each TTS chunk is written to the speaker.

        Records the chunk's RMS + timestamp so `should_interrupt()` can
        subtract the TTS envelope from the incoming mic envelope.
        """
        if pcm_int16.size == 0:
            return
        rms = float(np.sqrt(np.mean((pcm_int16.astype(np.float32) / 32768.0) ** 2)))
        now = time.time()
        self._tts_envelope.append((now, rms))
        # Trim to window
        cutoff = now - self._tts_env_window_s
        while self._tts_envelope and self._tts_envelope[0][0] < cutoff:
            self._tts_envelope.popleft()

    # ── main detector ───────────────────────────────────────────────────
    def should_interrupt(self, chunk: np.ndarray) -> bool:
        """Return True if the current mic chunk represents a genuine barge-in.

        `chunk` is float32 in [-1, 1], 16 kHz mono.
        """
        if chunk.size == 0:
            return False

        mic_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        tts_rms = self._current_tts_rms()

        # Envelope AEC-lite: assume the mic picks up a fraction of the TTS.
        corrected = max(0.0, mic_rms - self._tts_subtract_gain * tts_rms)

        # Is this a "speech-y" chunk per Silero?
        # The Silero VAD exposes the last computed probability on
        # `last_prob`; we don't want to trigger a full state-machine
        # step here because that would interfere with the normal
        # speech_start/speech_end callbacks used for turn boundaries.
        prob = float(getattr(self._vad, "last_prob", 0.0))
        vad_threshold = float(getattr(self._vad, "threshold", 0.5))
        is_speech = prob >= vad_threshold

        # Fallback to an energy test if the VAD hasn't been warmed up
        # yet (prob == 0 after a reset).
        if prob <= 0.0:
            is_speech = corrected > self._noise_floor * _db_to_ratio(self._snr_margin_db)

        # Update noise floor on non-speech chunks only.
        if not is_speech:
            self._noise_floor = (
                (1.0 - self._noise_alpha) * self._noise_floor
                + self._noise_alpha * corrected
            )
            self._speech_streak_start = None
            return False

        # SNR-gated speech — ignore quiet speech events.
        snr_ok = corrected > self._noise_floor * _db_to_ratio(self._snr_margin_db)
        if not snr_ok:
            self._speech_streak_start = None
            return False

        # Require sustained voiced speech.
        now = time.time()
        if self._speech_streak_start is None:
            self._speech_streak_start = now
            return False
        if (now - self._speech_streak_start) >= self._min_sustained_s:
            self._speech_streak_start = None  # reset for next event
            return True
        return False

    # ── introspection for tests / telemetry ─────────────────────────────
    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def _current_tts_rms(self) -> float:
        if not self._tts_envelope:
            return 0.0
        now = time.time()
        # Use the most recent sample within the window (the TTS is "loud"
        # right now if the last sample is recent; zero otherwise).
        ts, rms = self._tts_envelope[-1]
        if now - ts > 0.3:
            return 0.0
        return rms


def _db_to_ratio(db: float) -> float:
    return float(10.0 ** (db / 20.0))
