"""
Emotion pipeline orchestrator.

Camera → face detection → emotion classification → smoothed valence/arousal state.
Designed to run in its own thread so consumers (visualizer, LLM context, robot
actuators) can poll `get_state()` cheaply.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .camera import CSICamera
from .emotion_classifier import EMOTION_LABELS, EmotionClassifier
from .face_detector import FaceDetector

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    label: str = "Neutral"
    confidence: float = 0.0
    probs: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    valence: float = 0.0
    arousal: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    fps: float = 0.0
    latency_ms: float = 0.0
    has_face: bool = False
    frame: Optional[np.ndarray] = None  # latest BGR frame (small preview)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probs": {EMOTION_LABELS[i]: float(self.probs[i]) for i in range(len(EMOTION_LABELS))},
            "valence": self.valence,
            "arousal": self.arousal,
            "bbox": self.bbox,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "has_face": self.has_face,
        }


class EmotionPipeline:
    """
    Threaded camera → face → emotion loop.

    Usage:
        pipe = EmotionPipeline(config["vision"])
        pipe.start()
        ...
        state = pipe.get_state()
        ...
        pipe.stop()
    """

    def __init__(self, config: dict):
        self.cfg = config or {}
        self.camera = CSICamera(
            sensor_id=self.cfg.get("sensor_id", 0),
            width=self.cfg.get("width", 1280),
            height=self.cfg.get("height", 720),
            fps=self.cfg.get("fps", 30),
            flip_method=self.cfg.get("flip_method", 0),
            use_csi=self.cfg.get("use_csi", True),
        )
        self.face_detector = FaceDetector(
            model_path=self.cfg.get("face_model_path", "models/vision/face_detection_yunet_2023mar.onnx"),
            score_threshold=self.cfg.get("face_score_threshold", 0.6),
        )
        self.classifier = EmotionClassifier(
            model_path=self.cfg.get("emotion_model_path", "models/vision/enet_b0_8_best_afew.onnx"),
            prefer_gpu=True,
        )
        self.detect_every_n = max(1, int(self.cfg.get("detect_every_n_frames", 2)))
        self.smoothing = float(self.cfg.get("smoothing", 0.7))
        self.staleness_fade_s = float(self.cfg.get("staleness_fade_seconds", 3.0))

        self._state = EmotionState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_face_seen_at: float = 0.0

    # -------------------- thread control --------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("EmotionPipeline started")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.camera.close()
        logger.info("EmotionPipeline stopped")

    def get_state(self) -> EmotionState:
        with self._lock:
            return self._state

    # -------------------- face selection --------------------
    @staticmethod
    def _pick_face(
        faces: List[Tuple[int, int, int, int, float]],
        last_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        """Pick the face to track this frame.

        If we had a face last frame, prefer the one whose centre is closest
        to the previous bbox centre AND whose size is at least 40% of the
        previous size (rules out random background detections grabbing the
        track). Otherwise pick by a score(confidence) × size(area) product,
        so we don't hand the track to a low-confidence giant bbox.
        """
        if last_bbox is not None:
            lx, ly, lw, lh = last_bbox
            lcx = lx + lw / 2.0
            lcy = ly + lh / 2.0
            l_area = max(1, lw * lh)
            best = None
            best_d = None
            for (x, y, w, h, _score) in faces:
                if w * h < 0.4 * l_area and l_area > 1:
                    continue
                cx = x + w / 2.0
                cy = y + h / 2.0
                d = (cx - lcx) ** 2 + (cy - lcy) ** 2
                # Also skip jumps that are too large to be the same person
                if d > (max(lw, lh) * 3) ** 2:
                    continue
                if best_d is None or d < best_d:
                    best_d = d
                    best = (x, y, w, h)
            if best is not None:
                return best
        # First frame (or last track lost): score × area so a confident
        # mid-size face beats a low-confidence massive bbox.
        faces_ranked = sorted(faces, key=lambda f: f[4] * f[2] * f[3], reverse=True)
        x, y, w, h, _ = faces_ranked[0]
        return (x, y, w, h)

    # -------------------- main loop --------------------
    def _loop(self) -> None:
        frame_idx = 0
        fps_window: List[float] = []
        while self._running:
            t0 = time.perf_counter()
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Re-detect every N frames; reuse last bbox in between.
            if frame_idx % self.detect_every_n == 0:
                try:
                    faces = self.face_detector.detect(frame)
                except Exception as exc:
                    # Don't let a detector blow up the pipeline thread —
                    # log once, treat the frame as faceless, keep going.
                    logger.error(f"face_detector.detect crashed: {exc!r}")
                    faces = []
                if faces:
                    self._last_bbox = self._pick_face(faces, self._last_bbox)
                else:
                    self._last_bbox = None
            frame_idx += 1

            new_state = EmotionState(frame=frame)
            now = time.time()
            if self._last_bbox is not None:
                self._last_face_seen_at = now
                x, y, w, h = self._last_bbox
                # Expand bbox slightly so the classifier sees full face
                pad = int(0.1 * max(w, h))
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(frame.shape[1], x + w + pad)
                y1 = min(frame.shape[0], y + h + pad)
                face_crop = frame[y0:y1, x0:x1]
                if face_crop.size > 0:
                    probs, label, conf = self.classifier.classify(face_crop)
                    v, a = EmotionClassifier.valence_arousal(probs)

                    # EMA smoothing on valence/arousal + probs
                    with self._lock:
                        prev = self._state
                    s = self.smoothing
                    if prev.has_face:
                        v = s * prev.valence + (1 - s) * v
                        a = s * prev.arousal + (1 - s) * a
                        probs = s * prev.probs + (1 - s) * probs

                    new_state.has_face = True
                    new_state.label = label
                    new_state.confidence = conf
                    new_state.probs = probs.astype(np.float32)
                    new_state.valence = float(v)
                    new_state.arousal = float(a)
                    new_state.bbox = (x, y, w, h)
            else:
                # No face this frame — fade last valence/arousal toward neutral
                with self._lock:
                    prev = self._state
                since = now - self._last_face_seen_at if self._last_face_seen_at else 1e9
                fade = max(0.0, 1.0 - (since / max(0.1, self.staleness_fade_s)))
                new_state.has_face = False
                new_state.valence = prev.valence * fade
                new_state.arousal = prev.arousal * fade
                new_state.label = prev.label if fade > 0.05 else "Neutral"
                new_state.confidence = prev.confidence * fade
                new_state.probs = prev.probs * fade

            dt = time.perf_counter() - t0
            new_state.latency_ms = dt * 1000.0
            fps_window.append(dt)
            if len(fps_window) > 30:
                fps_window.pop(0)
            avg = sum(fps_window) / len(fps_window)
            new_state.fps = 1.0 / avg if avg > 0 else 0.0

            with self._lock:
                self._state = new_state

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
