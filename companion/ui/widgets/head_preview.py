"""Head pose preview widget — shows current pan/tilt as a rotating head.

Tries pyqtgraph.opengl for a 3D head (sphere + nose cone forward marker).
Falls back to a QPainter-based 2D top-down + side-view if 3D init fails —
Tegra (Jetson Orin Nano) GL drivers can be fussy, so this keeps the
calibration wizard usable regardless.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

log = logging.getLogger(__name__)


class HeadPreviewWidget(QWidget):
    """Live head-pose visualization. Emits nothing; purely a display."""

    pose_changed = pyqtSignal(float, float)   # pan_deg, tilt_deg (forwarded for hooks)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pan_deg = 0.0
        self._tilt_deg = 0.0
        self._target_pan = 0.0
        self._target_tilt = 0.0
        self._left_tick = 0
        self._right_tick = 0
        self._left_temp = 0.0
        self._right_temp = 0.0
        self._pan_limits = (-90.0, 90.0)
        self._tilt_limits = (-30.0, 30.0)

        self._use_3d = False
        self._gl_view = None
        self._head_mesh = None
        self._nose_mesh = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._viewport = self._build_3d_view() or self._build_2d_view()
        layout.addWidget(self._viewport, stretch=1)

        self._readout = QLabel("pan: 0.0°   tilt: 0.0°")
        self._readout.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setFamily("Monospace")
        font.setStyleHint(QFont.Monospace)
        self._readout.setFont(font)
        layout.addWidget(self._readout)

    # ── view builders ─────────────────────────────────────────────────────
    def _build_3d_view(self) -> Optional[QWidget]:
        try:
            import numpy as np
            import pyqtgraph.opengl as gl
        except Exception as e:
            log.info(f"3D preview unavailable ({e}); falling back to 2D")
            return None
        try:
            view = gl.GLViewWidget()
            view.opts["distance"] = 6.0
            view.opts["elevation"] = 15
            view.opts["azimuth"] = 45

            grid = gl.GLGridItem()
            grid.scale(1, 1, 1)
            view.addItem(grid)

            # Head — sphere
            sphere_md = gl.MeshData.sphere(rows=24, cols=24, radius=1.0)
            head = gl.GLMeshItem(
                meshdata=sphere_md,
                smooth=True,
                color=(0.85, 0.85, 0.95, 1.0),
                shader="shaded",
                glOptions="opaque",
            )
            view.addItem(head)
            self._head_mesh = head

            # Nose — cone pointing +X (forward), drawn as a small cylinder with
            # one radius = 0 (cone). pyqtgraph MeshData.cylinder supports this.
            nose_md = gl.MeshData.cylinder(
                rows=8, cols=12, radius=[0.25, 0.0], length=0.6
            )
            nose = gl.GLMeshItem(
                meshdata=nose_md,
                smooth=True,
                color=(1.0, 0.5, 0.3, 1.0),
                shader="shaded",
                glOptions="opaque",
            )
            # Orient cone along +X (cylinder length is along +Z by default)
            nose.rotate(90, 0, 1, 0)
            nose.translate(0.9, 0.0, 0.0)
            view.addItem(nose)
            self._nose_mesh = nose

            # Eyes — two small spheres
            eye_md = gl.MeshData.sphere(rows=12, cols=12, radius=0.12)
            for y in (-0.35, 0.35):
                eye = gl.GLMeshItem(
                    meshdata=eye_md, smooth=True,
                    color=(0.1, 0.1, 0.1, 1.0), shader="shaded", glOptions="opaque",
                )
                eye.translate(0.75, y, 0.25)
                view.addItem(eye)

            self._use_3d = True
            self._gl_view = view
            log.info("3D head preview initialized")
            return view
        except Exception as e:
            log.warning(f"3D head preview init failed: {e}; falling back to 2D")
            return None

    def _build_2d_view(self) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        self._top_view = _PanIndicator(self)
        self._side_view = _TiltIndicator(self)
        layout.addWidget(self._top_view, stretch=1)
        layout.addWidget(self._side_view, stretch=1)
        return holder

    # ── public setters ────────────────────────────────────────────────────
    def set_limits(self, pan_limits_deg: tuple[float, float],
                   tilt_limits_deg: tuple[float, float]) -> None:
        self._pan_limits = tuple(pan_limits_deg)
        self._tilt_limits = tuple(tilt_limits_deg)
        if not self._use_3d:
            self._top_view.set_limits(self._pan_limits)
            self._side_view.set_limits(self._tilt_limits)
            self._top_view.update()
            self._side_view.update()

    def set_pose(self, pan_deg: float, tilt_deg: float,
                 target_pan: Optional[float] = None,
                 target_tilt: Optional[float] = None) -> None:
        self._pan_deg = pan_deg
        self._tilt_deg = tilt_deg
        if target_pan is not None:
            self._target_pan = target_pan
        if target_tilt is not None:
            self._target_tilt = target_tilt
        self._refresh()
        self.pose_changed.emit(pan_deg, tilt_deg)

    def set_telemetry(self, left_tick: int, right_tick: int,
                      left_temp: float, right_temp: float) -> None:
        self._left_tick = left_tick
        self._right_tick = right_tick
        self._left_temp = left_temp
        self._right_temp = right_temp
        self._refresh_readout()

    # ── rendering ─────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        if self._use_3d and self._head_mesh is not None:
            # Reset head transform and reapply pitch then yaw.
            # Sign convention used by the kinematics: +pan = right, +tilt = up.
            # pyqtgraph's right-hand rotations around Y/Z produce the opposite
            # in the default camera orientation, so we negate.
            tilt = -self._tilt_deg
            pan = -self._pan_deg
            self._head_mesh.resetTransform()
            self._head_mesh.rotate(tilt, 0, 1, 0)                # pitch around Y
            self._head_mesh.rotate(pan, 0, 0, 1)                 # yaw around Z

            self._nose_mesh.resetTransform()
            self._nose_mesh.rotate(90, 0, 1, 0)
            self._nose_mesh.translate(0.9, 0.0, 0.0)
            self._nose_mesh.rotate(tilt, 0, 1, 0)
            self._nose_mesh.rotate(pan, 0, 0, 1)
        else:
            self._top_view.set_value(self._pan_deg, self._target_pan)
            self._side_view.set_value(self._tilt_deg, self._target_tilt)
            self._top_view.update()
            self._side_view.update()
        self._refresh_readout()

    def _refresh_readout(self) -> None:
        txt = (
            f"pan: {self._pan_deg:+6.1f}°  (→ {self._target_pan:+6.1f}°)    "
            f"tilt: {self._tilt_deg:+6.1f}°  (→ {self._target_tilt:+6.1f}°)"
        )
        if self._left_tick or self._right_tick:
            txt += (
                f"    L: {self._left_tick:4d}tk {self._left_temp:4.1f}°C    "
                f"R: {self._right_tick:4d}tk {self._right_temp:4.1f}°C"
            )
        self._readout.setText(txt)


# ── 2D fallback indicators ──────────────────────────────────────────────────

class _Indicator(QWidget):
    """Base for pan/tilt 2D indicators."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._target = 0.0
        self._lo = -90.0
        self._hi = 90.0
        self.setMinimumSize(140, 140)

    def set_value(self, value: float, target: float = 0.0) -> None:
        self._value = value
        self._target = target

    def set_limits(self, limits: tuple[float, float]) -> None:
        self._lo, self._hi = limits


class _PanIndicator(_Indicator):
    """Top-down view: arrow rotates around center for pan (yaw)."""
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 10

        # Background circle + limit arc
        p.setPen(QPen(QColor("#888"), 1))
        p.setBrush(QBrush(QColor("#1b1b22")))
        p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

        # Limit markers
        p.setPen(QPen(QColor("#c44"), 2))
        for deg in (self._lo, self._hi):
            a = math.radians(deg - 90)                 # 0° = up
            p.drawLine(int(cx), int(cy),
                       int(cx + r * math.cos(a)), int(cy + r * math.sin(a)))

        # Target (dim)
        a = math.radians(self._target - 90)
        p.setPen(QPen(QColor("#4a4"), 2, Qt.DashLine))
        p.drawLine(int(cx), int(cy),
                   int(cx + (r - 8) * math.cos(a)),
                   int(cy + (r - 8) * math.sin(a)))

        # Current pose (bright arrow)
        a = math.radians(self._value - 90)
        p.setPen(QPen(QColor("#8cf"), 3))
        tip_x = cx + (r - 5) * math.cos(a)
        tip_y = cy + (r - 5) * math.sin(a)
        p.drawLine(int(cx), int(cy), int(tip_x), int(tip_y))

        p.setPen(QColor("#ccc"))
        p.drawText(5, 15, f"PAN  {self._value:+.1f}°")


class _TiltIndicator(_Indicator):
    """Side view: arrow rotates horizontally for tilt (pitch)."""
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 10

        p.setPen(QPen(QColor("#888"), 1))
        p.setBrush(QBrush(QColor("#1b1b22")))
        p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

        p.setPen(QPen(QColor("#c44"), 2))
        for deg in (self._lo, self._hi):
            a = math.radians(-deg)                     # 0° = right (forward)
            p.drawLine(int(cx), int(cy),
                       int(cx + r * math.cos(a)), int(cy + r * math.sin(a)))

        a = math.radians(-self._target)
        p.setPen(QPen(QColor("#4a4"), 2, Qt.DashLine))
        p.drawLine(int(cx), int(cy),
                   int(cx + (r - 8) * math.cos(a)),
                   int(cy + (r - 8) * math.sin(a)))

        a = math.radians(-self._value)
        p.setPen(QPen(QColor("#8cf"), 3))
        p.drawLine(int(cx), int(cy),
                   int(cx + (r - 5) * math.cos(a)),
                   int(cy + (r - 5) * math.sin(a)))

        p.setPen(QColor("#ccc"))
        p.drawText(5, 15, f"TILT {self._value:+.1f}°")
