"""
HSEmotion classifier (EfficientNet-B0, 8 emotions) — ONNX with CUDA EP.

Model: enet_b0_8_best_afew.onnx — produced by HSEmotion / face-emotion-recognition
Output classes (alphabetical, AffectNet 8): Anger, Contempt, Disgust, Fear,
Happiness, Neutral, Sadness, Surprise.

Each class is anchored in the Russell valence-arousal circumplex; the final
(valence, arousal) point is the probability-weighted sum of those anchors,
which gives a smooth 2D position for visualization rather than a jumpy label.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

EMOTION_LABELS: List[str] = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

# (valence, arousal) anchors in [-1, 1]^2 — Russell circumplex.
EMOTION_VA: Dict[str, Tuple[float, float]] = {
    "Anger":     (-0.7,  0.7),
    "Contempt":  (-0.5,  0.2),
    "Disgust":   (-0.8,  0.3),
    "Fear":      (-0.6,  0.8),
    "Happiness": ( 0.85, 0.55),
    "Neutral":   ( 0.0,  0.0),
    "Sadness":   (-0.7, -0.5),
    "Surprise":  ( 0.4,  0.85),
}

# ImageNet normalization (HSEmotion EfficientNet expects this)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmotionClassifier:
    def __init__(self, model_path: str, prefer_gpu: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"HSEmotion model not found: {model_path}\n"
                f"Run: python3 scripts/download_vision_models.py"
            )
        # Local import so the rest of the module can be inspected w/o ORT installed.
        import onnxruntime as ort

        available = ort.get_available_providers()
        if prefer_gpu and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            if prefer_gpu:
                logger.warning(
                    "CUDAExecutionProvider not available — falling back to CPU. "
                    "Install onnxruntime-gpu for GPU acceleration."
                )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_opts, providers=providers)
        self._active_providers = self._session.get_providers()

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        # Shape is (1, 3, H, W) — figure out H, W (defaults to 224 if dynamic).
        shape = inp.shape
        self._h = int(shape[2]) if isinstance(shape[2], int) else 224
        self._w = int(shape[3]) if isinstance(shape[3], int) else 224

        logger.info(
            f"HSEmotion loaded: {model_path} | input {self._w}x{self._h} "
            f"| providers={self._active_providers}"
        )

    @property
    def providers(self) -> List[str]:
        return self._active_providers

    def info(self) -> None:
        print(f"HSEmotion providers (active): {self._active_providers}")
        print(f"Input size: {self._w}x{self._h}")

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = (face - _MEAN) / _STD
        face = np.transpose(face, (2, 0, 1))[None, ...]  # 1x3xHxW
        return np.ascontiguousarray(face, dtype=np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def classify(self, face_bgr: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Returns (probs[8], top_label, top_confidence).
        """
        x = self._preprocess(face_bgr)
        logits = self._session.run(None, {self._input_name: x})[0][0]
        probs = self._softmax(logits.astype(np.float32))
        idx = int(np.argmax(probs))
        return probs, EMOTION_LABELS[idx], float(probs[idx])

    @staticmethod
    def valence_arousal(probs: np.ndarray) -> Tuple[float, float]:
        """Probability-weighted (valence, arousal) on Russell's circumplex."""
        v = 0.0
        a = 0.0
        for i, label in enumerate(EMOTION_LABELS):
            pv, pa = EMOTION_VA[label]
            v += float(probs[i]) * pv
            a += float(probs[i]) * pa
        return v, a
