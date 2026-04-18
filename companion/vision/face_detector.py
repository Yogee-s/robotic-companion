"""Face detection using OpenCV's YuNet ONNX model.

YuNet (`cv2.FaceDetectorYN`, OpenCV ≥ 4.6) is a small, accurate CNN face
detector distributed as a 230 KB ONNX file. It returns confidence scores,
handles profile/turned faces better than Haar, and runs on CPU at
5-10 ms per 320×320 inference on a Jetson Orin Nano — comparable to Haar
while being noticeably more robust.

Model: `models/vision/face_detection_yunet_2023mar.onnx` (downloaded by
`scripts/download_models.py`).
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]  # x, y, w, h, score


class FaceDetector:
    """YuNet ONNX face detector with graceful Haar fallback."""

    def __init__(
        self,
        model_path: str = "models/vision/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        det_width: int = 320,
        det_height: int = 320,
    ) -> None:
        self._det_w = int(det_width)
        self._det_h = int(det_height)
        self._score_th = float(score_threshold)
        self._model_path = model_path
        self._yn: cv2.FaceDetectorYN | None = None
        self._haar: cv2.CascadeClassifier | None = None

        if os.path.exists(model_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yn = cv2.FaceDetectorYN.create(
                    model_path,
                    "",
                    (self._det_w, self._det_h),
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    top_k=top_k,
                )
                logger.info(
                    f"YuNet face detector loaded ({model_path}, "
                    f"{self._det_w}x{self._det_h}, score ≥ {score_threshold})"
                )
            except Exception as exc:
                logger.error(f"YuNet init failed: {exc!r} — falling back to Haar")
                self._yn = None

        if self._yn is None:
            # Last-resort Haar fallback so the pipeline still runs if ONNX is absent.
            self._haar = self._load_haar()
            logger.warning(
                "Using Haar cascade fallback (yunet model missing). "
                "Run scripts/download_models.py to get the YuNet ONNX."
            )

    # ── public API ───────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[BBox]:
        if self._yn is not None:
            return self._detect_yunet(frame)
        return self._detect_haar(frame)

    # ── YuNet path ───────────────────────────────────────────────────────
    def _detect_yunet(self, frame: np.ndarray) -> List[BBox]:
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (self._det_w, self._det_h), interpolation=cv2.INTER_LINEAR)
        assert self._yn is not None
        self._yn.setInputSize((self._det_w, self._det_h))
        _, faces = self._yn.detect(small)
        if faces is None or len(faces) == 0:
            return []
        sx = fw / float(self._det_w)
        sy = fh / float(self._det_h)
        out: List[BBox] = []
        for f in faces:
            x, y, w, h = f[:4]
            score = float(f[-1])
            if score < self._score_th:
                continue
            bx = max(0, int(x * sx))
            by = max(0, int(y * sy))
            bw = max(1, min(int(w * sx), fw - bx))
            bh = max(1, min(int(h * sy), fh - by))
            out.append((bx, by, bw, bh, score))
        return out

    # ── Haar fallback ────────────────────────────────────────────────────
    @staticmethod
    def _load_haar() -> cv2.CascadeClassifier | None:
        for d in (
            getattr(getattr(cv2, "data", None), "haarcascades", ""),
            "/usr/share/opencv4/haarcascades/",
            "/usr/share/opencv/haarcascades/",
        ):
            if not d:
                continue
            p = os.path.join(d, "haarcascade_frontalface_default.xml")
            if os.path.exists(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    return c
        return None

    def _detect_haar(self, frame: np.ndarray) -> List[BBox]:
        if self._haar is None:
            return []
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(gray, 1.15, 5, minSize=(48, 48))
        if len(faces) == 0:
            return []
        sx, sy = fw / 640.0, fh / 480.0
        return [
            (
                max(0, int(x * sx)),
                max(0, int(y * sy)),
                max(1, int(w * sx)),
                max(1, int(h * sy)),
                1.0,
            )
            for (x, y, w, h) in faces
        ]
