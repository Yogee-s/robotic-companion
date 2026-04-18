"""Animated state pill (idle / listening / thinking / speaking)."""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QLabel, QWidget

from companion.ui.theme import PALETTE


_STATE_COLORS = {
    "idle":       PALETTE["muted"],
    "listening":  PALETTE["accent"],
    "processing": PALETTE["warn"],
    "speaking":   PALETTE["success"],
    "sleep":      PALETTE["muted"],
}

_STATE_LABEL = {
    "idle":       "Idle",
    "listening":  "Listening",
    "processing": "Thinking",
    "speaking":   "Speaking",
    "sleep":      "Sleeping",
}


class StatusIndicator(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(36)
        self._state = "idle"
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(60)

    def set_state(self, state: str) -> None:
        self._state = state
        self.update()

    def _advance(self) -> None:
        self._phase = (self._phase + 0.1) % (2 * 3.14159)
        if self._state in ("listening", "processing", "speaking"):
            self.update()

    def paintEvent(self, _event) -> None:
        import math

        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        color = QColor(_STATE_COLORS.get(self._state, PALETTE["muted"]))
        w, h = self.width(), self.height()
        r = min(h - 10, 20)
        qp.setBrush(color)
        qp.setPen(Qt.NoPen)
        pulse = int(r + (math.sin(self._phase) * 3 if self._state != "idle" else 0))
        qp.drawEllipse(12, (h - pulse) // 2, pulse, pulse)
        qp.setPen(QColor(PALETTE["text"]))
        qp.drawText(12 + pulse + 10, h // 2 + 5, _STATE_LABEL.get(self._state, self._state))
