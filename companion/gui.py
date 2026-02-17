"""
PyQt5 GUI for the AI Companion.

Displays conversation history, DOA visualization, audio level,
system stats, and control panel.
"""

import logging
import math
import time
import threading
from typing import Optional

import numpy as np

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLabel, QPushButton, QSlider, QComboBox, QGroupBox,
        QProgressBar, QFrame, QSplitter, QSizePolicy, QGridLayout,
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QEvent, QObject
    from PyQt5.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont,
        QLinearGradient, QRadialGradient, QKeyEvent,
    )
    HAS_QT = True
except ImportError:
    HAS_QT = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


# =====================================================================
# DOA Polar Plot
# =====================================================================
class DOAPolarPlot(QWidget):
    """Polar plot showing Direction of Arrival."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._vad_active = False
        self._history = []
        self._max_history = 20
        self.setMinimumSize(200, 200)

    def set_angle(self, angle: int, vad_active: bool = False):
        self._angle = angle % 360
        self._vad_active = vad_active
        self._history.append(self._angle)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 20

        painter.fillRect(self.rect(), QColor(30, 30, 40))

        # Concentric rings
        pen = QPen(QColor(60, 60, 80), 1)
        painter.setPen(pen)
        for frac in [0.25, 0.5, 0.75, 1.0]:
            r = int(radius * frac)
            painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)

        # Cross axes
        painter.setPen(QPen(QColor(60, 60, 80), 1, Qt.DashLine))
        painter.drawLine(cx, cy - radius, cx, cy + radius)
        painter.drawLine(cx - radius, cy, cx + radius, cy)

        # Direction labels
        painter.setPen(QColor(150, 150, 170))
        painter.setFont(QFont("sans-serif", 9))
        painter.drawText(cx - 6, cy - radius - 5, "0°")
        painter.drawText(cx + radius + 5, cy + 4, "90°")
        painter.drawText(cx - 10, cy + radius + 15, "180°")
        painter.drawText(cx - radius - 25, cy + 4, "270°")

        # History trail
        for i, angle in enumerate(self._history):
            alpha = int(40 + (i / self._max_history) * 80)
            rad = math.radians(angle - 90)
            trail_r = int(radius * 0.7)
            tx = cx + int(trail_r * math.cos(rad))
            ty = cy + int(trail_r * math.sin(rad))
            painter.setBrush(QBrush(QColor(100, 150, 255, alpha)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(tx - 3, ty - 3, 6, 6)

        # Main DOA indicator
        rad = math.radians(self._angle - 90)
        dx = cx + int(radius * 0.75 * math.cos(rad))
        dy = cy + int(radius * 0.75 * math.sin(rad))

        if self._vad_active:
            glow = QRadialGradient(dx, dy, 25)
            glow.setColorAt(0.0, QColor(0, 200, 100, 120))
            glow.setColorAt(1.0, QColor(0, 200, 100, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(dx - 25, dy - 25, 50, 50)

        color = QColor(0, 220, 120) if self._vad_active else QColor(100, 180, 255)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.drawEllipse(dx - 8, dy - 8, 16, 16)
        painter.setPen(QPen(color, 2, Qt.DashLine))
        painter.drawLine(cx, cy, dx, dy)

        # Angle text
        painter.setPen(QColor(220, 220, 240))
        painter.setFont(QFont("sans-serif", 11, QFont.Bold))
        painter.drawText(cx - 15, cy + radius + 30, f"{self._angle}°")
        painter.end()


# =====================================================================
# Audio Level Meter
# =====================================================================
class AudioLevelMeter(QWidget):
    """Vertical VU-style meter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self._peak = 0.0
        self._peak_decay = 0.95
        self.setMinimumSize(30, 100)
        self.setMaximumWidth(40)

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))
        self._peak = max(self._level, self._peak * self._peak_decay)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bar_w, bar_h = w - 8, h - 16
        bar_x, bar_y = 4, 8

        painter.fillRect(self.rect(), QColor(30, 30, 40))
        painter.setPen(QPen(QColor(60, 60, 80), 1))
        painter.drawRect(bar_x, bar_y, bar_w, bar_h)

        level_h = int(bar_h * self._level)
        if level_h > 0:
            grad = QLinearGradient(0, bar_y + bar_h, 0, bar_y)
            grad.setColorAt(0.0, QColor(0, 200, 100))
            grad.setColorAt(0.6, QColor(200, 200, 0))
            grad.setColorAt(1.0, QColor(255, 50, 50))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawRect(bar_x + 1, bar_y + bar_h - level_h, bar_w - 1, level_h)

        peak_y = bar_y + bar_h - int(bar_h * self._peak)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(bar_x, peak_y, bar_x + bar_w, peak_y)
        painter.end()


# =====================================================================
# Status Indicator
# =====================================================================
class StatusIndicator(QWidget):
    """Animated banner showing current conversation state."""

    STATE_COLORS = {
        "idle": QColor(100, 100, 120),
        "listening": QColor(0, 200, 100),
        "processing": QColor(255, 180, 0),
        "speaking": QColor(100, 150, 255),
    }
    STATE_LABELS = {
        "idle": "Ready — Press SPACE",
        "listening": "Listening ...",
        "processing": "Thinking ...",
        "speaking": "Speaking ...",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "idle"
        self.setFixedHeight(40)

    def set_state(self, state: str):
        self._state = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        color = self.STATE_COLORS.get(self._state, QColor(100, 100, 120))
        painter.fillRect(self.rect(), QColor(color.red(), color.green(), color.blue(), 40))
        painter.fillRect(0, 0, 4, h, color)
        painter.setPen(color)
        painter.setFont(QFont("sans-serif", 13, QFont.Bold))
        label = self.STATE_LABELS.get(self._state, self._state)
        painter.drawText(16, 0, w - 16, h, Qt.AlignVCenter, label)
        painter.end()


# =====================================================================
# Main GUI Window
# =====================================================================
class CompanionGUI(QMainWindow):
    """Main application window."""

    # Thread-safe signals
    state_changed_signal = pyqtSignal(str)
    transcription_signal = pyqtSignal(str)
    response_signal = pyqtSignal(str)
    response_token_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    doa_signal = pyqtSignal(int, bool)

    def __init__(self, config: dict, conversation_manager=None, respeaker=None):
        super().__init__()

        self._config = config
        self._conversation = conversation_manager
        self._respeaker = respeaker
        self._audio_input = None
        self._current_response = ""

        gui_cfg = config.get("gui", {})
        self._font_size = gui_cfg.get("font_size", 14)
        self._show_system_monitor = gui_cfg.get("show_system_monitor", True)
        self._show_doa = gui_cfg.get("show_doa_visualization", True)

        self.setWindowTitle("AI Companion")
        self.resize(
            gui_cfg.get("window_width", 1024),
            gui_cfg.get("window_height", 768),
        )

        self._space_held = False

        self._build_ui()
        self._connect_signals()
        self._apply_theme()
        self._start_timers()

        # Install app-level event filter so SPACE works regardless of
        # which widget has focus (QTextEdit would otherwise swallow it).
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # Status banner
        self._status = StatusIndicator()
        root.addWidget(self._status)

        # Splitter: conversation | sidebar
        splitter = QSplitter(Qt.Horizontal)

        # -- Left: conversation --
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Conversation")
        lbl.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        left_l.addWidget(lbl)

        self._chat = QTextEdit()
        self._chat.setReadOnly(True)
        self._chat.setMinimumWidth(400)
        left_l.addWidget(self._chat)

        self._live_label = QLabel("Ready")
        self._live_label.setStyleSheet(
            "color: #666; font-style: italic; padding: 4px;"
        )
        left_l.addWidget(self._live_label)
        splitter.addWidget(left)

        # -- Right: viz + controls --
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        if self._show_doa:
            viz = QHBoxLayout()
            self._doa_plot = DOAPolarPlot()
            viz.addWidget(self._doa_plot)
            self._level_meter = AudioLevelMeter()
            viz.addWidget(self._level_meter)
            right_l.addLayout(viz)

        # Controls
        cg = QGroupBox("Controls")
        cl = QGridLayout()

        # Mode toggle: Push-to-Talk vs Continuous
        cl.addWidget(QLabel("Mode:"), 0, 0)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Push to Talk", "Continuous"])
        self._mode_combo.setCurrentText("Push to Talk")
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        cl.addWidget(self._mode_combo, 0, 1)

        # Spacebar hint (shown in PTT mode)
        self._ptt_hint = QLabel("Hold SPACE to talk")
        self._ptt_hint.setAlignment(Qt.AlignCenter)
        self._ptt_hint.setMinimumHeight(40)
        self._ptt_hint.setStyleSheet(
            "background-color: #313244; border: 1px solid #45475a; "
            "border-radius: 6px; font-size: 14px; font-weight: bold; "
            "color: #89b4fa; padding: 8px;"
        )
        cl.addWidget(self._ptt_hint, 1, 0, 1, 2)

        # Singlish toggle
        self._singlish_btn = QPushButton("Singlish: OFF")
        self._singlish_btn.setCheckable(True)
        self._singlish_btn.setMinimumHeight(36)
        self._singlish_btn.clicked.connect(self._on_singlish_toggled)
        cl.addWidget(self._singlish_btn, 2, 0, 1, 2)

        cl.addWidget(QLabel("Verbosity:"), 3, 0)
        self._verbosity_combo = QComboBox()
        self._verbosity_combo.addItems(["brief", "normal", "detailed"])
        self._verbosity_combo.setCurrentText("normal")
        self._verbosity_combo.currentTextChanged.connect(self._on_verbosity_changed)
        cl.addWidget(self._verbosity_combo, 3, 1)

        cl.addWidget(QLabel("Volume:"), 4, 0)
        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(80)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        cl.addWidget(self._volume_slider, 4, 1)

        self._clear_btn = QPushButton("Clear History")
        self._clear_btn.clicked.connect(self._on_clear)
        cl.addWidget(self._clear_btn, 5, 0, 1, 2)

        cg.setLayout(cl)
        right_l.addWidget(cg)

        # System monitor
        if self._show_system_monitor:
            mg = QGroupBox("System")
            ml = QGridLayout()

            self._cpu_bar = QProgressBar()
            self._cpu_bar.setFormat("CPU: %p%")
            ml.addWidget(self._cpu_bar, 0, 0)

            self._mem_bar = QProgressBar()
            self._mem_bar.setFormat("MEM: %p%")
            ml.addWidget(self._mem_bar, 0, 1)

            self._temp_label = QLabel("Temp: --°C")
            ml.addWidget(self._temp_label, 1, 0)

            self._gpu_label = QLabel("GPU: --%")
            ml.addWidget(self._gpu_label, 1, 1)

            self._model_label = QLabel("Model: loading...")
            self._model_label.setWordWrap(True)
            self._model_label.setStyleSheet("font-size: 11px; color: #888;")
            ml.addWidget(self._model_label, 2, 0, 1, 2)

            mg.setLayout(ml)
            right_l.addWidget(mg)

        right.setMaximumWidth(350)
        splitter.addWidget(right)
        splitter.setSizes([650, 350])
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.state_changed_signal.connect(self._on_state_changed)
        self.transcription_signal.connect(self._on_transcription)
        self.response_signal.connect(self._on_response)
        self.response_token_signal.connect(self._on_response_token)
        self.error_signal.connect(self._on_error)
        self.doa_signal.connect(self._on_doa_update)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Inter', 'Roboto', sans-serif;
            }
            QTextEdit {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
                selection-background-color: #45475a;
            }
            QGroupBox {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 16px;
                font-weight: bold;
                color: #a6adc8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
            QPushButton:pressed, QPushButton:checked {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QSlider::groove:horizontal {
                background: #313244;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QProgressBar {
                background-color: #313244;
                border: none;
                border-radius: 4px;
                text-align: center;
                color: #cdd6f4;
                font-size: 11px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 4px;
            }
            QLabel { color: #a6adc8; }
            QSplitter::handle {
                background-color: #313244;
                width: 2px;
            }
        """)

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def _start_timers(self):
        gui_cfg = self._config.get("gui", {})

        # System stats timer
        if self._show_system_monitor:
            self._sys_timer = QTimer(self)
            self._sys_timer.timeout.connect(self._update_system_stats)
            self._sys_timer.start(gui_cfg.get("system_monitor_interval_ms", 2000))

        # Audio level timer (replaces the bare daemon thread)
        if self._show_doa:
            self._level_timer = QTimer(self)
            self._level_timer.timeout.connect(self._update_audio_level)
            self._level_timer.start(50)  # 20 Hz

    # ------------------------------------------------------------------
    # Slots (thread-safe)
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_state_changed(self, state: str):
        self._status.set_state(state)
        styles = {
            "listening": ("Listening ...", "color: #00dd77; font-style: italic; padding: 4px;"),
            "processing": ("Thinking ...", "color: #ffbb00; font-style: italic; padding: 4px;"),
            "speaking": ("Speaking ...", "color: #88aaff; font-style: italic; padding: 4px;"),
        }
        if state == "idle" and self._conversation and self._conversation.mode == "ptt":
            text, style = ("Press SPACE to talk", "color: #89b4fa; font-style: italic; padding: 4px;")
        else:
            text, style = styles.get(state, ("Ready", "color: #666; font-style: italic; padding: 4px;"))
        self._live_label.setText(text)
        self._live_label.setStyleSheet(style)

    @pyqtSlot(str)
    def _on_transcription(self, text: str):
        self._append_msg("You", text, "#89b4fa")
        self._current_response = ""

    @pyqtSlot(str)
    def _on_response(self, text: str):
        self._append_msg("AI", text, "#a6e3a1")
        self._current_response = ""

    @pyqtSlot(str)
    def _on_response_token(self, token: str):
        self._current_response += token
        self._live_label.setText(self._current_response[-100:])

    @pyqtSlot(str)
    def _on_error(self, error: str):
        self._append_msg("System", f"Error: {error}", "#f38ba8")

    @pyqtSlot(int, bool)
    def _on_doa_update(self, angle: int, vad: bool):
        if self._show_doa:
            self._doa_plot.set_angle(angle, vad)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _append_msg(self, sender: str, text: str, color: str):
        ts = time.strftime("%H:%M:%S")
        html = (
            f'<div style="margin: 6px 0;">'
            f'<span style="color: {color}; font-weight: bold;">{sender}</span> '
            f'<span style="color: #585b70; font-size: 11px;">{ts}</span><br/>'
            f'<span style="color: #cdd6f4;">{text}</span></div>'
        )
        self._chat.append(html)
        sb = self._chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _update_audio_level(self):
        """Poll audio input level (called by QTimer at 20 Hz)."""
        if self._audio_input and self._show_doa:
            level = self._audio_input.get_level()
            self._level_meter.set_level(level)

    def _on_mode_changed(self, mode_text: str):
        if not self._conversation:
            return
        if mode_text == "Push to Talk":
            self._conversation.set_mode("ptt")
            self._ptt_hint.setText("Hold SPACE to talk")
            self._ptt_hint.setStyleSheet(
                "background-color: #313244; border: 1px solid #45475a; "
                "border-radius: 6px; font-size: 14px; font-weight: bold; "
                "color: #89b4fa; padding: 8px;"
            )
        else:
            self._conversation.set_mode("continuous")
            self._ptt_hint.setText("Listening continuously")
            self._ptt_hint.setStyleSheet(
                "background-color: #1a3a1a; border: 1px solid #2d5a2d; "
                "border-radius: 6px; font-size: 14px; font-weight: bold; "
                "color: #a6e3a1; padding: 8px;"
            )

    def _on_singlish_toggled(self):
        enabled = self._singlish_btn.isChecked()
        if self._conversation:
            self._conversation.set_singlish(enabled)
        if enabled:
            self._singlish_btn.setText("Singlish: ON lah!")
            self._singlish_btn.setStyleSheet(
                "background-color: #f38ba8; color: #1e1e2e; "
                "border: 1px solid #f38ba8; border-radius: 6px; "
                "font-size: 13px; font-weight: bold; padding: 8px;"
            )
        else:
            self._singlish_btn.setText("Singlish: OFF")
            self._singlish_btn.setStyleSheet("")

    def _on_verbosity_changed(self, level: str):
        if self._conversation:
            self._conversation.set_verbosity(level)

    def _on_volume_changed(self, value: int):
        if self._conversation and hasattr(self._conversation, "_audio_output"):
            self._conversation._audio_output.volume = value / 100.0

    def _on_clear(self):
        if self._conversation:
            self._conversation.clear_history()
        self._chat.clear()
        self._append_msg("System", "History cleared.", "#fab387")

    def _update_system_stats(self):
        if not HAS_PSUTIL:
            return
        try:
            self._cpu_bar.setValue(int(psutil.cpu_percent(interval=None)))
            self._mem_bar.setValue(int(psutil.virtual_memory().percent))

            # Temperature
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for entries in temps.values():
                        if entries:
                            self._temp_label.setText(f"Temp: {entries[0].current:.0f}°C")
                            break
                else:
                    with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
                        self._temp_label.setText(f"Temp: {int(f.read().strip()) / 1000:.0f}°C")
            except Exception:
                pass

            # GPU (Jetson-specific)
            try:
                with open("/sys/devices/gpu.0/load") as f:
                    self._gpu_label.setText(f"GPU: {int(f.read().strip()) / 10:.0f}%")
            except Exception:
                self._gpu_label.setText("GPU: N/A")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # External wiring
    # ------------------------------------------------------------------

    def connect_conversation(self, conv):
        """Wire conversation manager callbacks to Qt signals."""
        self._conversation = conv
        conv.on_state_changed = lambda s: self.state_changed_signal.emit(s)
        conv.on_transcription = lambda t: self.transcription_signal.emit(t)
        conv.on_response = lambda t: self.response_signal.emit(t)
        conv.on_response_token = lambda t: self.response_token_signal.emit(t)
        conv.on_error = lambda e: self.error_signal.emit(e)

    def connect_audio_input(self, audio_input):
        """Store audio input reference for level metering."""
        self._audio_input = audio_input

    def set_model_info(self, info: str):
        if self._show_system_monitor:
            self._model_label.setText(info)

    # ------------------------------------------------------------------
    # Spacebar → Push-to-Talk  (app-level event filter)
    # ------------------------------------------------------------------

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Intercept SPACE key at application level for push-to-talk."""
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat() and not self._space_held:
                if self._conversation and self._conversation.mode == "ptt":
                    self._space_held = True
                    self._conversation.push_to_talk_pressed()
                    self._ptt_hint.setText("Listening ...")
                    self._ptt_hint.setStyleSheet(
                        "background-color: #1a3a1a; border: 2px solid #00dd77; "
                        "border-radius: 6px; font-size: 14px; font-weight: bold; "
                        "color: #00dd77; padding: 8px;"
                    )
                    return True  # consume the event
        elif event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat() and self._space_held:
                if self._conversation and self._conversation.mode == "ptt":
                    self._space_held = False
                    self._conversation.push_to_talk_released()
                    self._ptt_hint.setText("Hold SPACE to talk")
                    self._ptt_hint.setStyleSheet(
                        "background-color: #313244; border: 1px solid #45475a; "
                        "border-radius: 6px; font-size: 14px; font-weight: bold; "
                        "color: #89b4fa; padding: 8px;"
                    )
                    return True  # consume the event
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        if self._conversation:
            self._conversation.stop()
        event.accept()
