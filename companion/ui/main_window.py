"""Main desktop window — three-panel layout (transcript · awareness · signals).

The window is a view onto the ConversationManager and subsystems. It
forwards user input (SPACE for PTT, mode combobox, clear history,
Singlish toggle) back to the manager via its public API, and receives
state updates through Qt signals that can be emitted from non-Qt threads.
"""

from __future__ import annotations

import logging

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QKeyEvent
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from companion.core.config import AppConfig
from companion.ui.theme import PALETTE
from companion.ui.widgets.doa_plot import DOAPolarPlot
from companion.ui.widgets.emotion_circumplex import EmotionCircumplex
from companion.ui.widgets.level_bars import RMSBar
from companion.ui.widgets.metric_card import MetricCard
from companion.ui.widgets.status_indicator import StatusIndicator
from companion.ui.widgets.system_monitor import SystemMonitor

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    state_signal = pyqtSignal(str)
    transcript_signal = pyqtSignal(str, str)
    doa_signal = pyqtSignal(float, bool)
    emotion_signal = pyqtSignal(float, float, str)
    scene_signal = pyqtSignal(str)

    def __init__(
        self,
        cfg: AppConfig,
        conversation,
        emotion_pipeline=None,
        scene_watcher=None,
        respeaker=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = conversation
        self.emotion = emotion_pipeline
        self.scene = scene_watcher
        self.respeaker = respeaker

        self.setWindowTitle("Companion")
        self.resize(cfg.gui.window_width, cfg.gui.window_height)

        self._build_menu()
        self._build_body()
        self._wire_conversation()

        self._viz_timer = QTimer(self)
        self._viz_timer.timeout.connect(self._pump)
        self._viz_timer.start(50)

        self.state_signal.connect(self._status.set_state)
        self.transcript_signal.connect(self._append_bubble)
        self.doa_signal.connect(self._doa.set_angle)
        self.emotion_signal.connect(lambda v, a, l: self._circumplex.set_state(v, a, l))
        self.scene_signal.connect(lambda c: self._scene_label.setText(c or "—"))

    # ── menu ─────────────────────────────────────────────────────────────
    def _build_menu(self) -> None:
        mb: QMenuBar = self.menuBar()
        view = mb.addMenu("View")
        view.addAction("Debug GUI…").triggered.connect(self._open_debug)

        model_menu: QMenu = mb.addMenu("Model")
        for key in self.cfg.llm.model_paths.keys():
            model_menu.addAction(key).triggered.connect(
                lambda _=False, k=key: log.info(f"Model switch to '{k}' — edit config.yaml and restart.")
            )

        settings = mb.addMenu("Settings")
        act_singlish = settings.addAction("Singlish")
        act_singlish.setCheckable(True)
        act_singlish.setChecked(self.conv.singlish)
        act_singlish.triggered.connect(lambda v: self.conv.set_singlish(v))

    # ── layout ───────────────────────────────────────────────────────────
    def _build_body(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        left = QFrame()
        left.setObjectName("Card")
        left.setMinimumWidth(400)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 14, 14)
        title = QLabel("Conversation")
        title.setObjectName("Heading")
        self._transcript = QTextBrowser()
        self._transcript.setOpenExternalLinks(True)
        self._transcript.setFont(QFont("Inter", 12))
        ctl = QHBoxLayout()
        self._ptt_btn = QPushButton("Hold SPACE to talk")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["ptt", "continuous", "wake_word"])
        self._mode_combo.setCurrentText(self.conv.mode)
        self._mode_combo.currentTextChanged.connect(self.conv.set_mode)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.conv.clear_history)
        ctl.addWidget(self._mode_combo)
        ctl.addWidget(self._ptt_btn, 1)
        ctl.addWidget(clear)
        ll.addWidget(title)
        ll.addWidget(self._transcript, 1)
        ll.addLayout(ctl)

        centre = QFrame()
        centre.setObjectName("Card")
        lc = QVBoxLayout(centre)
        lc.setContentsMargins(14, 14, 14, 14)
        lc.setSpacing(8)
        lc_title = QLabel("Awareness")
        lc_title.setObjectName("Heading")
        self._status = StatusIndicator()
        self._circumplex = EmotionCircumplex()
        self._scene_label = QLabel("—")
        self._scene_label.setObjectName("Subtle")
        self._scene_label.setWordWrap(True)
        self._scene_label.setAlignment(Qt.AlignCenter)
        lc.addWidget(lc_title)
        lc.addWidget(self._status)
        lc.addWidget(self._circumplex, 1)
        lc.addWidget(QLabel("Scene:"))
        lc.addWidget(self._scene_label)

        right = QFrame()
        right.setObjectName("Card")
        right.setFixedWidth(360)
        lr = QVBoxLayout(right)
        lr.setContentsMargins(14, 14, 14, 14)
        lr.setSpacing(10)
        lr.addWidget(self._heading("Signals"))
        self._doa = DOAPolarPlot()
        meter_row = QHBoxLayout()
        self._rms = RMSBar()
        meter_row.addWidget(QLabel("Mic"))
        meter_row.addWidget(self._rms)
        meter_row.addStretch(1)
        self._stt_card = MetricCard("STT", "—", "ms")
        self._llm_card = MetricCard("LLM", "—", "tok/s")
        cards = QHBoxLayout()
        cards.addWidget(self._stt_card)
        cards.addWidget(self._llm_card)
        lr.addWidget(self._doa)
        lr.addLayout(meter_row)
        lr.addLayout(cards)
        lr.addWidget(SystemMonitor(self.cfg.gui.system_monitor_interval_ms), 1)

        outer.addWidget(left, 4)
        outer.addWidget(centre, 3)
        outer.addWidget(right, 0)

    @staticmethod
    def _heading(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("Heading")
        return lbl

    # ── conversation callbacks (any thread) ─────────────────────────────
    def _wire_conversation(self) -> None:
        self.conv.on_state_changed = lambda s: self.state_signal.emit(s)
        self.conv.on_transcription = lambda t: self.transcript_signal.emit("user", t)
        self.conv.on_response = lambda t: self.transcript_signal.emit("assistant", t)

    def _append_bubble(self, role: str, text: str) -> None:
        color = PALETTE["accent"] if role == "user" else PALETTE["success"]
        speaker = "You" if role == "user" else "Companion"
        html = (
            f"<div style='margin:6px 0;'>"
            f"<b style='color:{color};'>{speaker}:</b> "
            f"<span style='color:{PALETTE['text']};'>{text}</span>"
            f"</div>"
        )
        self._transcript.append(html)

    # ── periodic polling ────────────────────────────────────────────────
    def _pump(self) -> None:
        if self.emotion is not None:
            try:
                st = self.emotion.get_state()
                self.emotion_signal.emit(st.valence, st.arousal, st.label)
            except Exception:
                pass
        if self.scene is not None:
            try:
                ss = self.scene.get_state()
                self.scene_signal.emit(ss.caption)
            except Exception:
                pass

    # ── keyboard ────────────────────────────────────────────────────────
    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
            self.conv.push_to_talk_pressed()
            self._ptt_btn.setText("Listening…")
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev: QKeyEvent) -> None:
        if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
            self.conv.push_to_talk_released()
            self._ptt_btn.setText("Hold SPACE to talk")
        super().keyReleaseEvent(ev)

    def _open_debug(self) -> None:
        try:
            from tests.debug_gui import launch

            launch(
                cfg=self.cfg,
                conversation=self.conv,
                emotion=self.emotion,
                respeaker=self.respeaker,
                scene=self.scene,
            )
        except Exception:
            log.exception("Failed to open debug GUI")
