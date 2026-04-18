"""Head-tracks a detected face by feeding the bbox offset from the frame
centre into a proportional head-pose controller.

Pipeline: EmotionPipeline.get_state() → face bbox → pixel error → degree error
→ HeadController.set_head_pose(target_pan, target_tilt). set_head_pose
already clamps to cfg.pan_limits_deg / cfg.tilt_limits_deg, so this module
cannot drive the head past any calibrated limit.

When no face is visible for face_lost_grace_s, the commanded pose slides
back toward (0, 0) at recenter_rate_deg_per_s.

Intended for Section 11 of the head_motor_quickstart notebook. Keep the
dependencies light — just numpy, opencv, and the two existing modules.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from companion.motor.controller import HeadController
from companion.vision.pipeline import EmotionPipeline, EmotionState

log = logging.getLogger(__name__)


@dataclass
class TrackerSnapshot:
    """Last-frame state exposed for rendering. Immutable per frame."""
    frame: Optional[np.ndarray] = None           # BGR copy of the frame the tracker just processed
    has_face: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    pan_err_deg: float = 0.0
    tilt_err_deg: float = 0.0
    target_pan_deg: float = 0.0
    target_tilt_deg: float = 0.0
    actual_pan_deg: float = 0.0
    actual_tilt_deg: float = 0.0
    recentering: bool = False
    t: float = field(default_factory=time.monotonic)


class FaceTracker:
    def __init__(
        self,
        head: HeadController,
        vision: EmotionPipeline,
        kp: float = 0.3,
        deadband_deg: float = 4.0,
        update_hz: float = 15.0,
        frame_width: int = 1280,
        frame_height: int = 720,
        camera_hfov_deg: float = 62.0,
        camera_vfov_deg: float = 37.0,
        face_lost_grace_s: float = 1.0,
        recenter_rate_deg_per_s: float = 8.0,
    ):
        self.head = head
        self.vision = vision
        self.kp = float(kp)
        self.deadband_deg = float(deadband_deg)
        self.update_hz = float(update_hz)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.camera_hfov_deg = float(camera_hfov_deg)
        self.camera_vfov_deg = float(camera_vfov_deg)
        self.face_lost_grace_s = float(face_lost_grace_s)
        self.recenter_rate_deg_per_s = float(recenter_rate_deg_per_s)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snap_lock = threading.Lock()
        self._snapshot: Optional[TrackerSnapshot] = None
        self._last_face_t: float = 0.0

    # ── public API ─────────────────────────────────────────────────────────
    def run(self, duration_s: Optional[float] = None) -> None:
        """Blocking control loop. Returns when duration elapses or stop() is called."""
        self._stop.clear()
        try:
            self._loop(duration_s)
        finally:
            self._safe_recenter()

    def start_async(self) -> None:
        """Run the control loop in a daemon thread. Use stop() to end."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(None,), name="FaceTracker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._safe_recenter()

    def latest_snapshot(self) -> Optional[TrackerSnapshot]:
        with self._snap_lock:
            return self._snapshot

    # ── loop ──────────────────────────────────────────────────────────────
    def _loop(self, duration_s: Optional[float]) -> None:
        period = 1.0 / max(1.0, self.update_hz)
        px_per_deg_x = self.frame_width / max(1e-6, self.camera_hfov_deg)
        px_per_deg_y = self.frame_height / max(1e-6, self.camera_vfov_deg)
        deg_per_px_x = 1.0 / px_per_deg_x
        deg_per_px_y = 1.0 / px_per_deg_y
        t_start = time.monotonic()

        while not self._stop.is_set():
            t0 = time.monotonic()
            if duration_s is not None and (t0 - t_start) >= duration_s:
                break

            state: EmotionState = self.vision.get_state()
            snap = TrackerSnapshot(
                actual_pan_deg=float(self.head.state.pan_deg),
                actual_tilt_deg=float(self.head.state.tilt_deg),
                target_pan_deg=float(self.head.state.target_pan_deg),
                target_tilt_deg=float(self.head.state.target_tilt_deg),
            )
            if state.frame is not None:
                snap.frame = state.frame.copy()

            now = time.monotonic()
            if state.has_face and state.bbox is not None:
                self._last_face_t = now
                x, y, w, h = state.bbox
                face_cx = x + w / 2.0
                face_cy = y + h / 2.0

                # Use the detector's source frame size if available, else fallback to config
                fw = state.frame.shape[1] if state.frame is not None else self.frame_width
                fh = state.frame.shape[0] if state.frame is not None else self.frame_height

                dx = face_cx - (fw / 2.0)
                dy = face_cy - (fh / 2.0)
                # Scale pixel→deg by the *actual* frame size in case it differs from config
                pan_err = dx * (self.camera_hfov_deg / fw)
                tilt_err = -dy * (self.camera_vfov_deg / fh)  # screen-Y is inverted

                snap.has_face = True
                snap.bbox = (x, y, w, h)
                snap.pan_err_deg = pan_err
                snap.tilt_err_deg = tilt_err

                # Per-axis soft deadband: each axis has its own "middle zone"
                # where the face can sit without triggering motion, and an
                # error on one axis doesn't drag the other axis along with it.
                # Soft (subtract the threshold) rather than hard (zero below,
                # full above) so the response is continuous — no jump right
                # outside the deadzone.
                eff_pan  = _soft_deadband(pan_err,  self.deadband_deg)
                eff_tilt = _soft_deadband(tilt_err, self.deadband_deg)

                if eff_pan != 0.0 or eff_tilt != 0.0:
                    # Integrate from last commanded target (already clamped).
                    new_pan  = self.head.state.target_pan_deg  + self.kp * eff_pan
                    new_tilt = self.head.state.target_tilt_deg + self.kp * eff_tilt
                    try:
                        self.head.set_head_pose(new_pan, new_tilt)
                    except Exception as e:
                        log.warning(f"set_head_pose failed: {e}")
                    # Refresh snapshot to reflect the post-clamp values
                    snap.target_pan_deg = float(self.head.state.target_pan_deg)
                    snap.target_tilt_deg = float(self.head.state.target_tilt_deg)
            else:
                # No face this frame. If it's been absent long enough, drift home.
                since = now - self._last_face_t if self._last_face_t else 1e9
                if since >= self.face_lost_grace_s:
                    snap.recentering = True
                    cur_pan = self.head.state.target_pan_deg
                    cur_tilt = self.head.state.target_tilt_deg
                    step = self.recenter_rate_deg_per_s * period
                    new_pan = _move_toward(cur_pan, 0.0, step)
                    new_tilt = _move_toward(cur_tilt, 0.0, step)
                    if new_pan != cur_pan or new_tilt != cur_tilt:
                        try:
                            self.head.set_head_pose(new_pan, new_tilt)
                        except Exception as e:
                            log.warning(f"set_head_pose (recenter) failed: {e}")
                        snap.target_pan_deg = float(self.head.state.target_pan_deg)
                        snap.target_tilt_deg = float(self.head.state.target_tilt_deg)

            with self._snap_lock:
                self._snapshot = snap

            # Sleep the remainder of the period
            elapsed = time.monotonic() - t0
            rem = period - elapsed
            if rem > 0:
                self._stop.wait(rem)

    def _safe_recenter(self) -> None:
        """Best-effort: command (0, 0) on exit so the head ends in a known pose."""
        try:
            self.head.set_head_pose(0.0, 0.0)
        except Exception as e:
            log.debug(f"on-exit recenter failed: {e}")


def _move_toward(x: float, target: float, max_step: float) -> float:
    if x == target:
        return x
    delta = target - x
    if abs(delta) <= max_step:
        return target
    return x + math.copysign(max_step, delta)


def _soft_deadband(err: float, db: float) -> float:
    """Soft deadband: 0 inside ±db, linear outside with the deadband value
    subtracted so the response is continuous at the boundary."""
    if abs(err) <= db:
        return 0.0
    return err - math.copysign(db, err)


# ── rendering helper ────────────────────────────────────────────────────────

def render_annotated_frame(
    snap: TrackerSnapshot,
    *,
    draw_crosshair: bool = True,
    draw_text: bool = True,
) -> Optional[np.ndarray]:
    """Return a BGR ndarray with tracker overlays drawn on the snapshot frame.
    None if the snapshot has no frame."""
    if snap is None or snap.frame is None:
        return None
    img = snap.frame.copy()
    fh, fw = img.shape[:2]

    if draw_crosshair:
        cx, cy = fw // 2, fh // 2
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (200, 200, 200), 1)
        cv2.line(img, (cx, cy - 12), (cx, cy + 12), (200, 200, 200), 1)

    if snap.has_face and snap.bbox is not None:
        x, y, w, h = snap.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 120), 2)
        # Line from frame centre to face centre
        cv2.line(
            img, (fw // 2, fh // 2),
            (int(x + w / 2), int(y + h / 2)),
            (0, 220, 120), 1,
        )

    if draw_text:
        lines = []
        if snap.has_face:
            lines.append(
                f"face err: pan={snap.pan_err_deg:+5.1f} deg  "
                f"tilt={snap.tilt_err_deg:+5.1f} deg"
            )
        else:
            lines.append("no face" + ("  (recentering)" if snap.recentering else ""))
        lines.append(
            f"target:   pan={snap.target_pan_deg:+5.1f} deg  "
            f"tilt={snap.target_tilt_deg:+5.1f} deg"
        )
        lines.append(
            f"actual:   pan={snap.actual_pan_deg:+5.1f} deg  "
            f"tilt={snap.actual_tilt_deg:+5.1f} deg"
        )
        for i, txt in enumerate(lines):
            y = 22 + 22 * i
            # outline for readability
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img
