"""Russell's valence-arousal circumplex with the current emotion + trail."""

from __future__ import annotations

import math
from collections import deque

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from companion.ui.theme import PALETTE

_ANCHORS = [
    ("Happy",     +0.7,  +0.5),
    ("Excited",   +0.4,  +0.8),
    ("Surprise",   0.0,  +0.9),
    ("Fear",      -0.4,  +0.8),
    ("Anger",     -0.6,  +0.6),
    ("Disgust",   -0.6,   0.0),
    ("Sad",       -0.6,  -0.4),
    ("Calm",      +0.5,  -0.4),
    ("Neutral",    0.0,   0.0),
]


class EmotionCircumplex(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self._v = 0.0
        self._a = 0.0
        self._label = "Neutral"
        self._trail: deque = deque(maxlen=40)

    def set_state(self, valence: float, arousal: float, label: str = "") -> None:
        self._v = float(valence)
        self._a = float(arousal)
        self._label = label or self._label
        self._trail.append((self._v, self._a))
        self.update()

    def paintEvent(self, _event) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(cx, cy) - 18

        qp.fillRect(self.rect(), QColor(PALETTE["bg"]))
        qp.setPen(QPen(QColor(PALETTE["surface2"]), 1))
        qp.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
        qp.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
        qp.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))

        # Axis labels
        qp.setPen(QPen(QColor(PALETTE["text_dim"]), 1))
        qp.drawText(int(cx + r - 24), int(cy + 14), "valence+")
        qp.drawText(int(cx - r + 2), int(cy + 14), "valence-")
        qp.drawText(int(cx + 4), int(cy - r + 10), "arousal+")
        qp.drawText(int(cx + 4), int(cy + r - 2), "arousal-")

        # Anchors
        for name, v, a in _ANCHORS:
            x = cx + v * r
            y = cy - a * r
            qp.setPen(QPen(QColor(PALETTE["muted"]), 1))
            qp.setBrush(QColor(PALETTE["surface"]))
            qp.drawEllipse(int(x - 3), int(y - 3), 6, 6)
            qp.setPen(QPen(QColor(PALETTE["text_dim"]), 1))
            qp.drawText(int(x + 6), int(y + 4), name)

        # Trail
        if self._trail:
            prev = None
            for i, (v, a) in enumerate(self._trail):
                x = cx + v * r
                y = cy - a * r
                alpha = int(200 * (i + 1) / len(self._trail))
                col = QColor(PALETTE["accent"])
                col.setAlpha(alpha)
                qp.setPen(QPen(col, 1))
                if prev is not None:
                    qp.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
                prev = (x, y)

        # Current dot
        x = cx + self._v * r
        y = cy - self._a * r
        qp.setPen(QPen(QColor(PALETTE["bg"]), 1))
        qp.setBrush(QColor(PALETTE["accent"]))
        qp.drawEllipse(int(x - 8), int(y - 8), 16, 16)
        qp.setPen(QPen(QColor(PALETTE["text"]), 1))
        qp.drawText(int(x + 10), int(y - 4), self._label)
