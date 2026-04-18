"""
CSI camera capture for Jetson via GStreamer (nvarguscamerasrc).

Threaded grabber: always exposes the latest frame, dropping stale ones so
downstream consumers (face detection, emotion classification) never lag behind.
Falls back to a V4L2 device (/dev/video0) if CSI capture fails.
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """Build a GStreamer pipeline string for the Jetson CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 max-buffers=2 sync=false"
    )


class CSICamera:
    """
    Threaded CSI camera grabber.

    Tries the GStreamer/nvarguscamerasrc pipeline first; falls back to a V4L2
    device if that fails (useful for USB webcams or non-Jetson dev machines).
    """

    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0,
        use_csi: bool = True,
    ):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.use_csi = use_csi

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._backend = "uninitialized"

        self._open()
        self._start()

    def _open(self) -> None:
        if self.use_csi:
            pipeline = gstreamer_pipeline(
                self.sensor_id, self.width, self.height, self.fps, self.flip_method
            )
            logger.info(f"Opening CSI camera (sensor-id={self.sensor_id}) via GStreamer")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self._cap = cap
                self._backend = "gstreamer/nvarguscamerasrc"
                logger.info(f"CSI camera open: {self.width}x{self.height} @ {self.fps}fps")
                return
            logger.warning("CSI/GStreamer open failed — falling back to V4L2")

        # V4L2 fallback
        cap = cv2.VideoCapture(self.sensor_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.sensor_id)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera (sensor_id={self.sensor_id}). "
                f"Check that the camera is connected and accessible."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._backend = "v4l2"
        logger.info(f"V4L2 camera open: device={self.sensor_id}")

    def _start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # IMX219/nvargus needs ~0.5-1s for auto-exposure to converge; early
        # frames come out green/magenta from the ISP. Wait for the first frame,
        # then discard a warm-up window of frames so the first one returned is
        # correctly exposed.
        for _ in range(100):
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.02)
        else:
            logger.warning("No frame received within 2s of camera startup")
            return
        time.sleep(1.2)  # AE/AWB settle
        with self._lock:
            self._frame = None  # force consumer to see a post-settle frame

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame (BGR), or None if not yet available."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def backend(self) -> str:
        return self._backend

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
