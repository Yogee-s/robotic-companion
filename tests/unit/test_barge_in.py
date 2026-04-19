"""Adaptive barge-in — suppresses self-echo + door-slam transients."""

import numpy as np

from companion.audio.barge_in import BargeInDetector


class _FakeVAD:
    """Stub VAD — report a fixed probability per call sequence."""

    def __init__(self, probs):
        self._probs = list(probs)
        self.threshold = 0.5
        self.last_prob = 0.0

    def _advance(self):
        self.last_prob = self._probs.pop(0) if self._probs else 0.0


def test_silence_updates_noise_floor():
    vad = _FakeVAD(probs=[0.1, 0.1, 0.1])
    det = BargeInDetector(vad, min_sustained_ms=0)
    silence = np.zeros(512, dtype=np.float32)
    for _ in range(3):
        vad._advance()
        assert det.should_interrupt(silence) is False
    # Noise floor stays small on silence.
    assert det.noise_floor < 0.01


def test_speech_triggers_only_after_sustained_window():
    vad = _FakeVAD(probs=[0.9] * 10)
    det = BargeInDetector(vad, min_sustained_ms=100)
    loud = 0.3 * np.random.randn(512).astype(np.float32)
    vad._advance()
    # First chunk starts the streak but shouldn't fire yet.
    first = det.should_interrupt(loud)
    # Subsequent chunks fire once the min-sustained window is reached.
    import time
    time.sleep(0.15)
    vad._advance()
    second = det.should_interrupt(loud)
    assert first is False
    assert second is True
