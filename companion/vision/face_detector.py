"""Face detection via YOLO26n-pose.

One detector, no fallback chain. The pose model is loaded via onnxruntime
(GPU when CUDAExecutionProvider is available) and the face bbox is built
from the nose + eyes + ears keypoints — much more robust on wide-angle
cameras than generic face detectors, because the network localises the
whole body first and only then do we extract the head.

Model path defaults to `models/vision/yolo26n-pose.onnx`. If the file is
missing, construction fails with a clear error — better than silently
falling back to a worse detector.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]  # x, y, w, h, score

# COCO keypoint indices for face bbox synthesis (YOLO pose output layout)
_KPT_NOSE = 0
_KPT_LEYE = 1
_KPT_REYE = 2
_KPT_LEAR = 3
_KPT_REAR = 4
_FACE_KPT_IDS = (_KPT_NOSE, _KPT_LEYE, _KPT_REYE, _KPT_LEAR, _KPT_REAR)


class FaceDetector:
    """YOLO26n-pose face detector. Builds a face bbox from head keypoints."""

    def __init__(
        self,
        model_path: str = "models/vision/yolo26n-pose.onnx",
        score_threshold: float = 0.5,
        kpt_visibility_threshold: float = 0.25,
        input_size: int = 640,
        # Kept for backwards compat with EmotionPipeline's call site — unused.
        nms_threshold: float = 0.0,
        top_k: int = 0,
        det_width: int = 0,
        det_height: int = 0,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO pose model not found at {model_path}. "
                f"Export it with `yolo export model=yolo26n-pose.pt format=onnx` "
                f"or place the file at that path."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required; `pip install onnxruntime-gpu` "
                "on Jetson for GPU inference."
            ) from exc

        providers = ort.get_available_providers()
        want = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers
            else ["CPUExecutionProvider"]
        )
        self._sess = ort.InferenceSession(model_path, providers=want)
        self._input_name = self._sess.get_inputs()[0].name
        self._input_size = int(input_size)
        self._score_th = float(score_threshold)
        self._kpt_vis_th = float(kpt_visibility_threshold)
        logger.info(
            f"FaceDetector loaded {model_path} "
            f"(providers={self._sess.get_providers()}, score ≥ {score_threshold})"
        )

    # ── public API ───────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[BBox]:
        img, scale, pad = self._letterbox(frame)
        nchw = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self._sess.run(None, {self._input_name: nchw})[0]        # [1, 300, 57]
        return self._postprocess(out[0], frame.shape[:2], scale, pad)

    # ── preprocessing ────────────────────────────────────────────────────
    def _letterbox(self, frame: np.ndarray):
        """Resize-with-padding to input_size × input_size, preserving aspect.
        Returns (rgb_letterboxed, scale, (pad_x, pad_y))."""
        fh, fw = frame.shape[:2]
        s = self._input_size
        scale = min(s / fw, s / fh)
        new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_x = (s - new_w) // 2
        pad_y = (s - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas, scale, (pad_x, pad_y)

    # ── postprocessing ───────────────────────────────────────────────────
    def _postprocess(
        self,
        detections: np.ndarray,                # [300, 57]
        frame_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[BBox]:
        fh, fw = frame_shape
        pad_x, pad_y = pad
        out: List[BBox] = []
        confs = detections[:, 4]
        keep = confs >= self._score_th
        if not keep.any():
            return out
        dets = detections[keep]
        for det in dets:
            score = float(det[4])
            kpts = det[6:6 + 17 * 3].reshape(17, 3)           # 17 kpts × (x, y, vis)
            head = kpts[list(_FACE_KPT_IDS)]
            vis = head[:, 2] >= self._kpt_vis_th
            if vis.sum() < 2:
                continue
            xs_all = (head[vis, 0] - pad_x) / scale
            ys_all = (head[vis, 1] - pad_y) / scale

            # Prefer the midpoint between the two eyes as the bbox centre —
            # the tracker uses bbox centre as its target, so this makes the
            # robot aim at the user's eyes regardless of head tilt. Falls
            # back to the average of whichever head keypoints ARE visible.
            le_vis = head[_KPT_LEYE, 2] >= self._kpt_vis_th
            re_vis = head[_KPT_REYE, 2] >= self._kpt_vis_th
            if le_vis and re_vis:
                le = ((head[_KPT_LEYE, 0] - pad_x) / scale,
                      (head[_KPT_LEYE, 1] - pad_y) / scale)
                re = ((head[_KPT_REYE, 0] - pad_x) / scale,
                      (head[_KPT_REYE, 1] - pad_y) / scale)
                cx, cy = (le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0
            else:
                cx = float(xs_all.mean())
                cy = float(ys_all.mean())

            # Size: use the span of the visible head keypoints, expanded to
            # approximate the full face bbox (~65% width, ~55% height span
            # of ear-to-ear / brow-to-chin).
            kw = max(1.0, float(xs_all.max() - xs_all.min()))
            kh = max(1.0, float(ys_all.max() - ys_all.min()))
            side = max(kw / 0.65, kh / 0.55)
            bx = max(0, int(cx - side / 2))
            by = max(0, int(cy - side / 2))
            bw = max(1, min(int(side), fw - bx))
            bh = max(1, min(int(side), fh - by))
            out.append((bx, by, bw, bh, score))
        return out
