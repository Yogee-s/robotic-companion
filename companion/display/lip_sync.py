"""Lip sync — uses Rhubarb Lip Sync if installed, otherwise derives visemes
from audio envelope.

Rhubarb is a free offline tool that maps audio to phoneme/viseme timings:
  https://github.com/DanielSWolf/rhubarb-lip-sync

If the `rhubarb` binary is on PATH we shell out to it. Otherwise we use a
lightweight envelope-based approximation that opens the mouth roughly in
sync with the waveform — not as accurate but zero setup.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

VISEMES = ("rest", "ahh", "oh", "ee", "mm", "fv", "l", "eh")


@dataclass
class VisemeEvent:
    start_s: float
    viseme: str


def _rhubarb_available() -> bool:
    try:
        subprocess.run(
            ["rhubarb", "--version"], capture_output=True, timeout=3.0, check=False
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def visemes_from_pcm(
    pcm_int16: bytes, sample_rate: int
) -> list[VisemeEvent]:
    """Return a sorted list of (time, viseme) events covering the audio."""
    if not pcm_int16:
        return [VisemeEvent(0.0, "rest")]
    if _rhubarb_available():
        try:
            return _rhubarb(pcm_int16, sample_rate)
        except Exception as exc:
            log.debug(f"Rhubarb failed, falling back to envelope: {exc!r}")
    return _envelope(pcm_int16, sample_rate)


# ─── Rhubarb path ────────────────────────────────────────────────────────────

def _rhubarb(pcm_int16: bytes, sample_rate: int) -> list[VisemeEvent]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
        path = wav.name
    try:
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(sample_rate)
            fh.writeframes(pcm_int16)
        out = subprocess.run(
            ["rhubarb", "-f", "json", "-q", path],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    data = json.loads(out.stdout)
    events: list[VisemeEvent] = []
    letter_to_viseme = {
        "A": "ahh", "B": "mm", "C": "eh", "D": "eh",
        "E": "oh", "F": "fv", "G": "l", "H": "rest", "X": "rest",
    }
    for cue in data.get("mouthCues", []):
        v = letter_to_viseme.get(cue.get("value", "X"), "rest")
        events.append(VisemeEvent(start_s=float(cue["start"]), viseme=v))
    return events or [VisemeEvent(0.0, "rest")]


# ─── Envelope fallback ───────────────────────────────────────────────────────

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
