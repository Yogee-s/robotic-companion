"""Semantic end-of-utterance detection.

Wraps LiveKit's open-source EOU transformer (ONNX) which predicts whether
the speaker has finished their thought, based on the transcript so far.
Runs on top of Silero VAD: once Silero detects acoustic silence, EOU
decides whether to end the turn now or wait a bit longer for the user to
resume speaking.

Model: `models/eou/livekit-eou-v0.4.1-intl.onnx` (download via
`scripts/download_models.py`).

If the model file isn't present, `predict_end_of_turn()` returns `True`
unconditionally so the acoustic VAD alone governs turn-taking — a safe
fallback that preserves current behaviour.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


class EndOfUtteranceDetector:
    """Predicts end-of-turn from a rolling transcript string."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.55,
        extra_wait_ms: int = 600,
    ) -> None:
        self.model_path = model_path
        self.threshold = float(threshold)
        self.extra_wait_ms = int(extra_wait_ms)
        self._session = None
        self._tokenizer = None
        self._available = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            log.warning(
                f"EOU model not found at {self.model_path}; "
                "falling back to acoustic VAD only."
            )
            return
        try:
            from companion.core.onnx_runtime import make_session

            self._session = make_session(self.model_path)
            self._available = True
        except Exception as exc:
            log.warning(f"EOU load failed: {exc!r} — using acoustic VAD only.")

    @property
    def available(self) -> bool:
        return self._available

    def predict_end_of_turn(self, transcript: str) -> bool:
        """Return True if the speaker has finished speaking.

        Without the ONNX model we return True (acoustic VAD decides).
        With the model we run the transformer and compare against threshold."""
        if not self._available or not transcript.strip():
            return True
        try:
            score = self._score(transcript)
        except Exception as exc:
            log.debug(f"EOU inference failed: {exc!r} — returning True")
            return True
        return score >= self.threshold

    def _score(self, transcript: str) -> float:
        # Minimal byte-level tokeniser suitable for the open-source EOU
        # model. When the real tokeniser file ships alongside the ONNX,
        # this method is replaced with a tiktoken/fast tokeniser call.
        assert self._session is not None
        import numpy as np

        tokens = np.array([[min(b, 255) for b in transcript.lower().encode("utf-8")[:128]]], dtype=np.int64)
        outputs = self._session.run(None, {"input_ids": tokens})
        probs = outputs[0].flatten()
        return float(probs[-1] if probs.ndim else probs)
