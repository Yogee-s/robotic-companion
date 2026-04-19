"""Runnable face-tracking demo.

Callable two ways:
  - `run(...)` from Python / Jupyter
  - `python -m companion.vision.face_track_demo [--args ...]` from the shell

Auto-detects whether it's running inside a Jupyter kernel and renders the
annotated camera feed accordingly (inline Image via IPython.display, or a
cv2.imshow window when standalone).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import logging
import sys
import time
from typing import Optional

import cv2

from companion.core.config import load_config
from companion.motor.controller import HeadController
# Force-reload the vision modules so Jupyter kernels pick up disk edits
# without a kernel restart. Cheap; safe when nothing has changed.
for _name in (
    "companion.vision.face_detector",
    "companion.vision.pipeline",
    "companion.vision.face_tracker",
):
    if _name in sys.modules:
        importlib.reload(sys.modules[_name])
from companion.vision.face_tracker import FaceTracker, render_annotated_frame
from companion.vision.pipeline import EmotionPipeline

log = logging.getLogger(__name__)


def _in_notebook() -> bool:
    """True if running inside a Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython                        # type: ignore
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def run(
    *,
    sim: bool = True,
    kp: float = 0.3,
    deadband: float = 4.0,
    hfov_deg: float = 62.0,
    vfov_deg: float = 37.0,
    update_hz: float = 15.0,
    duration_s: float = 20.0,
    display: str = "auto",                   # 'auto' | 'notebook' | 'window' | 'none'
    config_path: str = "config.yaml",
    display_hz: float = 10.0,
    display_max_width: int = 640,
) -> None:
    """Run the face-tracking control loop for `duration_s` seconds.

    In sim mode (default) no servos are driven — useful for a safe dry-run
    against the real camera. In live mode the head physically follows the
    detected face; soft limits from the calibrated config.yaml are enforced
    inside `HeadController.set_head_pose`.
    """
    if display == "auto":
        display = "notebook" if _in_notebook() else "window"

    app_cfg = load_config(config_path)
    motor_cfg = copy.deepcopy(app_cfg.motor)
    motor_cfg.sim_only = bool(sim)

    vision_cfg = (
        app_cfg.vision.__dict__ if hasattr(app_cfg.vision, "__dict__")
        else dict(app_cfg.vision)
    )
    vision = EmotionPipeline(vision_cfg)
    vision.start()
    time.sleep(1.0)                           # let camera + detector warm up

    head = HeadController(motor_cfg)
    head.connect()
    head.enable_torque(True)

    tracker = FaceTracker(
        head=head, vision=vision,
        kp=kp, deadband_deg=deadband,
        update_hz=update_hz,
        camera_hfov_deg=hfov_deg, camera_vfov_deg=vfov_deg,
    )
    tracker.start_async()

    mode_label = "SIM motors" if sim else "REAL motors"
    print(f"face_track_demo.run: {mode_label}, kp={kp}, deadband={deadband}°, "
          f"hfov={hfov_deg}°, duration={duration_s}s, display={display}")
    try:
        _display_loop(tracker, duration_s, display, display_hz, display_max_width)
    finally:
        print("stopping tracker, recentering, disabling torque…")
        tracker.stop()
        head.disconnect()
        vision.stop()
        if display == "window":
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("clean shutdown.")


def _display_loop(
    tracker: FaceTracker,
    duration_s: float,
    display: str,
    display_hz: float,
    display_max_width: int,
) -> None:
    period = 1.0 / max(1.0, display_hz)
    t_end = time.monotonic() + max(0.0, duration_s)

    # Lazy notebook imports so 'window' mode doesn't require IPython.
    ipy_display = ipy_image = ipy_clear = None
    if display == "notebook":
        from IPython.display import Image as _Image, clear_output as _clear, display as _display
        ipy_image, ipy_clear, ipy_display = _Image, _clear, _display

    window_name = "face_track_demo"
    window_created = False
    was_visible = False                           # set True once the window has actually drawn

    while time.monotonic() < t_end:
        t0 = time.monotonic()
        snap = tracker.latest_snapshot()
        img = render_annotated_frame(snap) if snap is not None else None
        if img is not None:
            h, w = img.shape[:2]
            if w > display_max_width:
                img = cv2.resize(img, (display_max_width, int(display_max_width * h / w)))
            if display == "notebook":
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    ipy_clear(wait=True)
                    ipy_display(ipy_image(data=jpg.tobytes()))
            elif display == "window":
                try:
                    if not window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, img.shape[1], img.shape[0])
                        window_created = True
                        print(f"window: opened ({img.shape[1]}x{img.shape[0]}). "
                              f"Press 'q' in the window or close it to stop.")
                    cv2.imshow(window_name, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("window: 'q' pressed, stopping.")
                        break
                    # X-button close detection — but only after the window has
                    # been visible at least once. WND_PROP_VISIBLE returns <1
                    # for both "closed" and "not yet drawn", so we need a
                    # one-shot edge: went visible, now invisible → closed.
                    visible = cv2.getWindowProperty(window_name,
                                                    cv2.WND_PROP_VISIBLE) >= 1
                    if visible:
                        was_visible = True
                    elif was_visible:
                        print("window: closed by user, stopping.")
                        break
                except cv2.error as exc:
                    print(f"window display failed ({exc}); is a DISPLAY configured? "
                          f"Fall back by passing display='notebook' or 'none'.")
                    break
        rem = period - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)


def _main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim", action="store_true",
                   help="use the simulated bus (no servos driven)")
    p.add_argument("--kp", type=float, default=0.3)
    p.add_argument("--deadband", type=float, default=4.0,
                   help="degrees; central dead zone per axis where the head stays still")
    p.add_argument("--hfov", type=float, default=62.0,
                   help="camera horizontal field of view in degrees")
    p.add_argument("--vfov", type=float, default=37.0)
    p.add_argument("--update-hz", type=float, default=15.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--display", choices=["auto", "notebook", "window", "none"],
                   default="auto")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(
        sim=args.sim, kp=args.kp, deadband=args.deadband,
        hfov_deg=args.hfov, vfov_deg=args.vfov,
        update_hz=args.update_hz, duration_s=args.duration,
        display=args.display, config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
