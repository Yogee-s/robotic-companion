"""RMS / probability / generic horizontal or vertical level bars."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressBar, QWidget

from companion.ui.theme import PALETTE


class RMSBar(QProgressBar):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOrientation(Qt.Vertical)
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(False)
        self.setFixedWidth(14)
        self.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid {PALETTE["surface2"]};
                border-radius: 4px;
                background: {PALETTE["bg"]};
            }}
            QProgressBar::chunk {{
                background-color: {PALETTE["success"]};
                border-radius: 3px;
            }}
            """
        )

    def set_rms(self, rms_0_1: float) -> None:
        self.setValue(int(max(0.0, min(1.0, rms_0_1)) * 100))


class ProbabilityBar(QProgressBar):
    def __init__(self, label: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setRange(0, 100)
        self.setValue(0)
        self.setFormat(f"{label}  %p%")
        self.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid {PALETTE["surface2"]};
                border-radius: 4px;
                background: {PALETTE["bg"]};
                text-align: center;
                color: {PALETTE["text"]};
            }}
            QProgressBar::chunk {{
                background-color: {PALETTE["accent"]};
                border-radius: 3px;
            }}
            """
        )

    def set_prob(self, p: float) -> None:
        self.setValue(int(max(0.0, min(1.0, p)) * 100))
