"""CPU / RAM / GPU / temperature readout — Jetson-aware via tegrastats hints
and /sys thermal zones, fallback to psutil for portability."""

from __future__ import annotations

import os
from typing import Optional

import psutil
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QGridLayout, QWidget

from companion.ui.widgets.metric_card import MetricCard


class SystemMonitor(QWidget):
    def __init__(self, interval_ms: int = 2000, parent=None) -> None:
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._cpu = MetricCard("CPU", "—", "%")
        self._mem = MetricCard("RAM", "—", "%")
        self._gpu = MetricCard("GPU", "—", "%")
        self._temp = MetricCard("Temp", "—", "°C")
        lay.addWidget(self._cpu, 0, 0)
        lay.addWidget(self._mem, 0, 1)
        lay.addWidget(self._gpu, 1, 0)
        lay.addWidget(self._temp, 1, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)
        self._tick()

    def _tick(self) -> None:
        try:
            self._cpu.set_value(f"{psutil.cpu_percent():.0f}")
            self._mem.set_value(f"{psutil.virtual_memory().percent:.0f}")
        except Exception:
            pass
        gpu = self._read_gpu()
        if gpu is not None:
            self._gpu.set_value(f"{gpu:.0f}")
        temp = self._read_temp()
        if temp is not None:
            self._temp.set_value(f"{temp:.0f}")

    @staticmethod
    def _read_gpu() -> Optional[float]:
        for path in (
            "/sys/devices/gpu.0/load",
            "/sys/devices/platform/gpu.0/load",
            "/sys/devices/17000000.gv11b/load",
        ):
            if os.path.exists(path):
                try:
                    with open(path) as fh:
                        val = fh.read().strip()
                    return float(val) / 10.0
                except Exception:
                    pass
        return None

    @staticmethod
    def _read_temp() -> Optional[float]:
        base = "/sys/class/thermal"
        if not os.path.isdir(base):
            return None
        readings = []
        for entry in os.listdir(base):
            if not entry.startswith("thermal_zone"):
                continue
            try:
                with open(os.path.join(base, entry, "temp")) as fh:
                    readings.append(float(fh.read().strip()) / 1000.0)
            except Exception:
                pass
        return max(readings) if readings else None
