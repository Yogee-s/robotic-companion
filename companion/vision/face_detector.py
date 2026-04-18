"""Face detection — MediaPipe primary, YuNet secondary, Haar last-resort.

Detector preference order:

1. **MediaPipe Face Detection** (`mediapipe.solutions.face_detection`).
   Self-contained TFLite, model_selection=1 reaches ~5 m. ~10 ms/frame on
   Jetson CPU. Primary path when `mediapipe` is installed.
2. **YuNet** (`cv2.FaceDetectorYN`, OpenCV ≥ 4.6). 230 KB ONNX. Works if
   mediapipe is missing and OpenCV is new enough. Falls back automatically
   if the system's OpenCV is too old for the 2023mar model.
3. **Haar cascade**. Always-available last resort. Weaker at distance but
   at least keeps the pipeline alive.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]  # x, y, w, h, score


class FaceDetector:
    """Face detector with MediaPipe → YuNet → Haar cascade preference order."""

    def __init__(
        self,
        model_path: str = "models/vision/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        det_width: int = 320,
        det_height: int = 320,
        mp_model_selection: int = 1,          # 0 = short range (~2 m), 1 = full range (~5 m)
    ) -> None:
        self._det_w = int(det_width)
        self._det_h = int(det_height)
        self._score_th = float(score_threshold)
        self._model_path = model_path
        self._mp: object = None                 # MediaPipe FaceDetection instance
        self._yn: Optional[cv2.FaceDetectorYN] = None
        self._haar: Optional[cv2.CascadeClassifier] = None

        # 1. MediaPipe (preferred when available — best at distant faces)
        try:
            import mediapipe as mp
            self._mp = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=score_threshold,
                model_selection=mp_model_selection,
            )
            logger.info(
                f"MediaPipe face detector loaded "
                f"(model_selection={mp_model_selection}, score ≥ {score_threshold})"
            )
        except ImportError:
            logger.info("mediapipe not installed — trying YuNet")
        except Exception as exc:
            logger.warning(f"MediaPipe init failed: {exc!r} — trying YuNet")

        # 2. YuNet (OpenCV's own — works if OpenCV ≥ 4.6 AND model present)
        if self._mp is None and os.path.exists(model_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yn = cv2.FaceDetectorYN.create(
                    model_path, "", (self._det_w, self._det_h),
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

        # 3. Haar cascade (always as last-resort)
        if self._mp is None and self._yn is None:
            self._haar = self._load_haar()
            logger.warning(
                "Using Haar cascade fallback (no mediapipe, no YuNet). "
                "For distant faces, `pip install mediapipe`."
            )

    # ── public API ───────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[BBox]:
        if self._mp is not None:
            try:
                return self._detect_mediapipe(frame)
            except Exception as exc:
                logger.error(
                    f"MediaPipe runtime failure ({exc!r}); dropping to YuNet/Haar."
                )
                self._mp = None
                if self._haar is None:
                    self._haar = self._load_haar()
        if self._yn is not None:
            try:
                return self._detect_yunet(frame)
            except cv2.error as exc:
                # YuNet ONNX 2023mar requires OpenCV ≥ 4.7. On older runtimes
                # the detector constructs fine but .detect() raises at first
                # frame. Drop to Haar permanently.
                logger.error(
                    f"YuNet runtime failure ({exc}); permanently falling back "
                    f"to Haar. Install mediapipe for a better alternative."
                )
                self._yn = None
                if self._haar is None:
                    self._haar = self._load_haar()
        return self._detect_haar(frame)

    # ── MediaPipe path ───────────────────────────────────────────────────
    # Geometry-sanity filters — reject detections that can't be real faces.
    # Real face bboxes are roughly square (0.6–1.6 aspect) and span at least
    # a handful of pixels. Large background boxes are typically very wide
    # or very tall, and spurious detections tend to be tiny.
    _MIN_ASPECT = 0.55
    _MAX_ASPECT = 1.8
    _MIN_FACE_PX = 24              # smaller than this is almost always noise
    _MAX_FACE_FRAC = 0.85          # faces bigger than 85% of the frame side → implausible

    def _detect_mediapipe(self, frame: np.ndarray) -> List[BBox]:
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp.process(rgb)             # type: ignore[attr-defined]
        dets = getattr(results, "detections", None)
        if not dets:
            return []
        max_dim = max(fw, fh) * self._MAX_FACE_FRAC
        out: List[BBox] = []
        for d in dets:
            score = float(d.score[0]) if getattr(d, "score", None) else 0.0
            if score < self._score_th:
                continue
            rb = d.location_data.relative_bounding_box
            bx = max(0, int(rb.xmin * fw))
            by = max(0, int(rb.ymin * fh))
            bw = max(1, min(int(rb.width * fw), fw - bx))
            bh = max(1, min(int(rb.height * fh), fh - by))
            # Size sanity
            if bw < self._MIN_FACE_PX or bh < self._MIN_FACE_PX:
                continue
            if bw > max_dim or bh > max_dim:
                continue
            # Aspect sanity
            aspect = bw / float(bh)
            if aspect < self._MIN_ASPECT or aspect > self._MAX_ASPECT:
                continue
            out.append((bx, by, bw, bh, score))
        return out

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
        """Haar detection tuned to also catch distant (small) faces.

        Runs at 960×540 instead of 640×480, with minSize=(30, 30) and
        scaleFactor=1.1 / minNeighbors=3 — sensitive enough that a face
        ~1.5–2 m from the camera is still detected, at the cost of slightly
        more false positives. The pipeline picks the largest face so false
        positives rarely "win" over a real face.
        """
        if self._haar is None:
            return []
        fh, fw = frame.shape[:2]
        work_w, work_h = 960, 540
        small = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,       # finer scale pyramid → smaller faces caught
            minNeighbors=3,        # lower = more sensitive
            minSize=(30, 30),      # ~5% of frame width — catches ~1.5–2 m faces
            maxSize=(work_w // 2, work_h // 2),
        )
        if len(faces) == 0:
            return []
        sx, sy = fw / float(work_w), fh / float(work_h)
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
