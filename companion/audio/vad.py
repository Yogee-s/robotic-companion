"""
Voice Activity Detection using Silero VAD (ONNX, no torch dependency).

Runs the Silero ONNX model through our shared onnxruntime factory on
CUDA. No torchaudio, no torch — the Jetson's torchaudio wheel links
against CUDA 13 / libtorch mismatched with what's installed, so
anything that touches torch at import time crashes. We feed PCM chunks
directly to the ONNX session and run its LSTM state ourselves.

Model: `models/vad/silero_vad.onnx`. Downloaded on first use (~2 MB)
if missing.

State machine: IDLE → SPEECH_START → SPEAKING → SPEECH_END → IDLE.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np

logger = logging.getLogger(__name__)

_SILERO_ONNX_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
# Expected chunk size for Silero v5 ONNX: 512 samples at 16 kHz = 32 ms.
_SILERO_CHUNK_SAMPLES = 512


class VADState:
    IDLE = "idle"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"


class VoiceActivityDetector:
    """
    Real-time VAD using Silero ONNX (via onnxruntime CUDA).

    State machine: IDLE → SPEECH_START → SPEAKING → SPEECH_END → IDLE
    """

    def __init__(self, config: dict):
        self.threshold = config.get("threshold", 0.5)
        self.silence_duration_ms = config.get("silence_duration_ms", 600)
        self.min_speech_duration_ms = config.get("min_speech_duration_ms", 250)
        self.sample_rate = 16000

        self._state = VADState.IDLE
        self._session = None          # onnxruntime.InferenceSession
        self._hidden_state: Optional[np.ndarray] = None
        self._speech_buffer: list[np.ndarray] = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._chunk_ms = 32
        self._chunk_buffer = np.zeros(0, dtype=np.float32)  # residual-carry for odd-sized chunks

        # Latest computed speech probability (0..1). Exposed for UI/debug.
        self.last_prob: float = 0.0

        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_vad_probability = None

        self._load_model()
        self._reset_state()

    def _load_model(self):
        """Load Silero ONNX via shared onnxruntime factory (no torch)."""
        from companion.core.onnx_runtime import make_session

        here = Path(__file__).resolve().parents[2]
        model_path = here / "models" / "vad" / "silero_vad.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            logger.info("Silero VAD ONNX not found — downloading (~2 MB)…")
            try:
                urlretrieve(_SILERO_ONNX_URL, str(model_path))
            except Exception as exc:
                logger.error("Silero download failed: %r — energy VAD fallback.", exc)
                return

        try:
            start = time.time()
            self._session = make_session(str(model_path), prefer_gpu=True)
            logger.info("Silero VAD (ONNX) loaded in %.1fs", time.time() - start)
        except Exception as exc:
            logger.error("Silero ONNX load failed: %r — energy VAD fallback.", exc)
            self._session = None

    def _reset_state(self) -> None:
        """Zero the model's LSTM hidden state. Call at turn boundaries."""
        # Silero v5 ONNX exposes a single combined state tensor (2, 1, 128).
        self._hidden_state = np.zeros((2, 1, 128), dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Process one audio chunk through the VAD state machine.

        Args:
            audio_chunk: float32 numpy array at 16 kHz.

        Returns:
            Current state string.
        """
        self._chunk_ms = len(audio_chunk) / self.sample_rate * 1000

        # Get speech probability
        prob = self._get_probability(audio_chunk)
        self.last_prob = float(prob)

        if self.on_vad_probability:
            try:
                self.on_vad_probability(prob)
            except Exception:
                pass

        is_speech = prob >= self.threshold

        # State machine
        if self._state == VADState.IDLE:
            if is_speech:
                self._state = VADState.SPEECH_START
                self._speech_buffer = [audio_chunk.copy()]
                self._speech_ms = self._chunk_ms
                self._silence_ms = 0
                if self.on_speech_start:
                    try:
                        self.on_speech_start()
                    except Exception:
                        pass

        elif self._state in (VADState.SPEECH_START, VADState.SPEAKING):
            self._speech_buffer.append(audio_chunk.copy())
            self._state = VADState.SPEAKING

            if is_speech:
                self._silence_ms = 0
                self._speech_ms += self._chunk_ms
            else:
                self._silence_ms += self._chunk_ms
                if self._silence_ms >= self.silence_duration_ms:
                    self._state = VADState.SPEECH_END

                    if self._speech_ms >= self.min_speech_duration_ms:
                        full_audio = np.concatenate(self._speech_buffer)
                        if self.on_speech_end:
                            try:
                                self.on_speech_end(full_audio)
                            except Exception as e:
                                logger.error(f"on_speech_end error: {e}")
                    else:
                        logger.debug(
                            f"Speech too short ({self._speech_ms:.0f}ms), ignored."
                        )

                    self._speech_buffer = []
                    self._speech_ms = 0
                    self._silence_ms = 0
                    self._state = VADState.IDLE

        return self._state

    def _get_probability(self, audio_chunk: np.ndarray) -> float:
        """Silero expects exactly 512 samples per inference at 16 kHz.

        We buffer whatever the caller gives us, run Silero on every full
        512-sample window, and return the last window's probability.
        Residual samples (e.g. the tail of a 513-sample input) carry
        over to the next call.
        """
        if self._session is None:
            # Energy-based fallback — keeps the robot functional if the
            # ONNX session failed to load at startup.
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
            return min(1.0, max(0.0, (rms - 0.005) * 10))

        # Concatenate residual + new samples, then consume 512-sample windows.
        buf = np.concatenate([self._chunk_buffer, audio_chunk.astype(np.float32)])
        prob = self.last_prob  # keep prior if no full window fits
        while buf.size >= _SILERO_CHUNK_SAMPLES:
            window = buf[:_SILERO_CHUNK_SAMPLES].reshape(1, -1)
            try:
                out, new_state = self._session.run(
                    None,
                    {
                        "input": window,
                        "state": self._hidden_state,
                        "sr": np.array(self.sample_rate, dtype=np.int64),
                    },
                )
                prob = float(out.flatten()[0])
                self._hidden_state = new_state
            except Exception as exc:
                logger.debug("Silero inference failed: %r", exc)
                break
            buf = buf[_SILERO_CHUNK_SAMPLES:]
        self._chunk_buffer = buf
        return prob

    def reset(self):
        self._state = VADState.IDLE
        self._speech_buffer = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._chunk_buffer = np.zeros(0, dtype=np.float32)
        self._reset_state()

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_speaking(self) -> bool:
        return self._state in (VADState.SPEECH_START, VADState.SPEAKING)
