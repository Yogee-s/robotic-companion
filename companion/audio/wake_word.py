"""Wake-word detector — openWakeWord backend.

Idle mode: this detector listens continuously on the mic while the main
pipeline stays quiet. When the wake phrase (default "Hey Buddy") is
heard, it fires `on_wake()` and the conversation manager engages.

openWakeWord ships pretrained models such as "hey_mycroft" and
"hey_jarvis" for quick bring-up; a custom "Hey Buddy" model can be trained
via openWakeWord's synthetic-data pipeline and dropped into
`models/wake_word/hey_buddy.tflite`.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(
        self,
        model_path: str,
        phrase: str = "hey buddy",
        sensitivity: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.phrase = phrase
        self.sensitivity = float(sensitivity)
        self._model = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            from openwakeword.model import Model  # type: ignore
        except ImportError:
            log.warning("openwakeword not installed — wake word disabled.")
            return

        model_paths: list[str] = []
        if os.path.exists(self.model_path):
            model_paths.append(self.model_path)

        try:
            self._model = Model(
                wakeword_models=model_paths if model_paths else None,
                inference_framework="onnx",
            )
            log.info(
                f"Wake word ready (phrase='{self.phrase}', "
                f"custom model: {'yes' if model_paths else 'no — using built-ins'})"
            )
        except Exception as exc:
            log.warning(f"Wake word load failed: {exc!r}")
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def process(self, pcm_int16: np.ndarray) -> float:
        """Feed a chunk of int16 PCM at 16 kHz. Returns best-match probability."""
        if self._model is None:
            return 0.0
        with self._lock:
            preds = self._model.predict(pcm_int16)
        if not preds:
            return 0.0
        return float(max(preds.values()))

    def is_triggered(self, pcm_int16: np.ndarray) -> bool:
        return self.process(pcm_int16) >= self.sensitivity
