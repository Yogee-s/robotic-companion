"""Lip sync — derives visemes from the audio envelope.

Maps PCM audio to mouth-shape events (visemes) so the face display can
animate lips roughly in sync with speech. Uses an energy-envelope
heuristic that classifies each 40 ms window into rest / mm / eh / ahh
based on RMS amplitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

VISEMES = ("rest", "ahh", "oh", "ee", "mm", "fv", "l", "eh")


@dataclass
class VisemeEvent:
    start_s: float
    viseme: str


def visemes_from_pcm(
    pcm_int16: bytes, sample_rate: int
) -> list[VisemeEvent]:
    """Return a sorted list of (time, viseme) events covering the audio."""
    if not pcm_int16:
        return [VisemeEvent(0.0, "rest")]
    return _envelope(pcm_int16, sample_rate)


# ─── Envelope-based viseme extraction ────────────────────────────────────────

def _envelope(pcm_int16: bytes, sample_rate: int) -> list[VisemeEvent]:
    audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return [VisemeEvent(0.0, "rest")]
    hop = max(1, int(sample_rate * 0.04))  # 25 fps viseme updates
    events: list[VisemeEvent] = []
    last_v = None
    for i in range(0, audio.size, hop):
        seg = np.abs(audio[i : i + hop])
        rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        if rms < 0.03:
            v = "rest"
        elif rms < 0.10:
            v = "mm"
        elif rms < 0.20:
            v = "eh"
        else:
            v = "ahh"
        if v != last_v:
            events.append(VisemeEvent(i / sample_rate, v))
            last_v = v
    return events or [VisemeEvent(0.0, "rest")]
