"""Live scrolling audio waveform, ~2 s buffer."""

from __future__ import annotations

import numpy as np

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from companion.ui.theme import PALETTE


class WaveformView(QWidget):
    def __init__(self, window_s: float = 2.0, sample_rate: int = 16000, parent=None) -> None:
        super().__init__(parent)
        self._sr = int(sample_rate)
        self._window = int(window_s * self._sr)
        self._buf = np.zeros(self._window, dtype=np.float32)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        if pg is None:
            self._plot = None
            lay.addWidget(QLabel("Install pyqtgraph to see waveforms.", alignment=Qt.AlignCenter))
            return

        pg.setConfigOption("background", PALETTE["bg"])
        pg.setConfigOption("foreground", PALETTE["text_dim"])
        self._plot = pg.PlotWidget()
        self._plot.setYRange(-1.0, 1.0, padding=0)
        self._plot.showGrid(x=False, y=False)
        self._plot.setMouseEnabled(False, False)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._curve = self._plot.plot(pen=pg.mkPen(color=PALETTE["accent"], width=1))
        lay.addWidget(self._plot)

    def push(self, chunk: np.ndarray) -> None:
        if self._plot is None:
            return
        n = len(chunk)
        if n >= self._window:
            self._buf = chunk[-self._window:].astype(np.float32)
        else:
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = chunk.astype(np.float32)
        self._curve.setData(self._buf)
