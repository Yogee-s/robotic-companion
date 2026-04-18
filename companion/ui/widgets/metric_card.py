"""Big-number metric card — used everywhere for latency / FPS / token rate."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

from companion.ui.theme import PALETTE


class MetricCard(QFrame):
    def __init__(self, label: str, value: str = "—", unit: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumSize(120, 80)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(2)

        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px;")

        row = QLabel()
        self._value = QLabel(value)
        self._value.setStyleSheet(f"color: {PALETTE['text']}; font-size: 22px; font-weight: 600;")
        self._unit = QLabel(unit)
        self._unit.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px;")
        self._unit.setAlignment(Qt.AlignBottom)

        from PyQt5.QtWidgets import QHBoxLayout

        row_w = QFrame()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(4)
        row_l.addWidget(self._value)
        row_l.addWidget(self._unit)
        row_l.addStretch(1)

        lay.addWidget(self._label)
        lay.addWidget(row_w)
        lay.addStretch(1)

    def set_value(self, value: str, unit: str = None) -> None:
        self._value.setText(value)
        if unit is not None:
            self._unit.setText(unit)
