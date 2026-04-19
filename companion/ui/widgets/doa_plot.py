"""ReSpeaker DOA polar plot — compass + trail."""

from __future__ import annotations

import math

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from companion.ui.theme import PALETTE


class DOAPolarPlot(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self._angle_deg = 0.0
        self._trail: list[float] = []
        self._active = False
        self._head_pan_deg: float | None = None  # optional secondary marker

    def set_angle(self, angle_deg: float, active: bool = True) -> None:
        self._angle_deg = float(angle_deg)
        self._active = active
        self._trail.append(angle_deg)
        if len(self._trail) > 30:
            self._trail.pop(0)
        self.update()

    def set_head_pan(self, pan_deg: float | None) -> None:
        """Overlay the robot's current head pan as a secondary marker.
        Pass None to hide it. Same convention as DOA (0=front, +right, -left)."""
        self._head_pan_deg = None if pan_deg is None else float(pan_deg)
        self.update()

    def clear_trail(self) -> None:
        """Wipe the historical trail — call after calibration so the plot
        reflects only post-calibration directions."""
        self._trail.clear()
        self.update()

    def paintEvent(self, _event) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(cx, cy) - 4

        # Background
        qp.fillRect(self.rect(), QColor(PALETTE["bg"]))
        qp.setPen(QPen(QColor(PALETTE["surface2"]), 1))
        for frac in (0.33, 0.66, 1.0):
            rr = int(r * frac)
            qp.drawEllipse(int(cx - rr), int(cy - rr), rr * 2, rr * 2)

        # Crosshair
        qp.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
        qp.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))

        # Trail
        qp.setPen(QPen(QColor(PALETTE["muted"]), 1))
        for i, ang in enumerate(self._trail):
            a = math.radians(ang - 90)
            x = cx + math.cos(a) * r * 0.85
            y = cy + math.sin(a) * r * 0.85
            alpha = int(255 * (i + 1) / len(self._trail) * 0.4)
            col = QColor(PALETTE["accent"])
            col.setAlpha(alpha)
            qp.setBrush(col)
            qp.drawEllipse(int(x - 2), int(y - 2), 4, 4)

        # Current DOA direction
        a = math.radians(self._angle_deg - 90)
        x = cx + math.cos(a) * r * 0.9
        y = cy + math.sin(a) * r * 0.9
        accent = QColor(PALETTE["accent" if self._active else "muted"])
        qp.setPen(QPen(accent, 3))
        qp.drawLine(int(cx), int(cy), int(x), int(y))
        qp.setBrush(accent)
        qp.drawEllipse(int(x - 5), int(y - 5), 10, 10)

        # Head pan marker (secondary) — drawn as a dashed arrow + ring tick.
        if self._head_pan_deg is not None:
            ah = math.radians(self._head_pan_deg - 90)
            hx = cx + math.cos(ah) * r * 0.9
            hy = cy + math.sin(ah) * r * 0.9
            warn = QColor(PALETTE["warn"])
            pen = QPen(warn, 2, Qt.DashLine)
            qp.setPen(pen)
            qp.drawLine(int(cx), int(cy), int(hx), int(hy))
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(warn, 2))
            qp.drawEllipse(int(hx - 6), int(hy - 6), 12, 12)
