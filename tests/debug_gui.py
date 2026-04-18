"""Debug GUI — single PyQt5 window with a tab per subsystem.

Reuses the same shared widgets (metric card, waveform, emotion
circumplex, DOA plot) as the main window so everything looks like one
family. Each tab is focused, self-contained, and drives a single
subsystem end-to-end.

Launch standalone:
    python3 -m tests.debug_gui

Or from the main app's View → Debug GUI menu.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtCore import Qt, QTimer, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from companion.core.config import AppConfig, load_config  # noqa: E402
from companion.core.logging import setup_logging  # noqa: E402
from companion.ui.theme import PALETTE, apply_theme  # noqa: E402
from companion.ui.widgets.doa_plot import DOAPolarPlot  # noqa: E402
from companion.ui.widgets.emotion_circumplex import EmotionCircumplex  # noqa: E402
from companion.ui.widgets.level_bars import RMSBar  # noqa: E402
from companion.ui.widgets.metric_card import MetricCard  # noqa: E402
from companion.ui.widgets.waveform_view import WaveformView  # noqa: E402

log = logging.getLogger(__name__)


# ── Base tab helper ─────────────────────────────────────────────────────────

def _two_col(left: QWidget, right: QWidget, left_w: int = 260) -> QWidget:
    outer = QWidget()
    lay = QHBoxLayout(outer)
    lay.setContentsMargins(14, 14, 14, 14)
    lay.setSpacing(12)
    left.setMinimumWidth(left_w)
    left.setMaximumWidth(left_w)
    lay.addWidget(left)
    lay.addWidget(right, 1)
    return outer


# ── Audio tab ───────────────────────────────────────────────────────────────

class AudioTab(QWidget):
    chunk_signal = pyqtSignal(object, float, float, float)  # chunk, rms, doa, vad

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._ai = None
        self._rs = None
        self._vad = None
        self._running = False

        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        ll.addWidget(QLabel("Audio in + DOA"))
        ll.addWidget(self._start_btn)
        ll.addWidget(self._stop_btn)
        ll.addStretch(1)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right)
        lr.setContentsMargins(12, 12, 12, 12)
        self._wv = WaveformView(sample_rate=cfg.audio.sample_rate)
        row = QHBoxLayout()
        self._rms = RMSBar()
        self._doa = DOAPolarPlot()
        self._doa_label = QLabel("DOA: —")
        row.addWidget(self._rms)
        self._vad_card = MetricCard("VAD prob", "0.00")
        row.addWidget(self._vad_card)
        row.addStretch(1)
        row.addWidget(self._doa_label)
        lr.addWidget(self._wv, 1)
        lr.addLayout(row)
        lr.addWidget(self._doa, 1)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0)
        outer.addWidget(right, 1)

        self.chunk_signal.connect(self._on_chunk)

    def _start(self) -> None:
        from companion.audio.io import AudioInput
        from companion.audio.respeaker import ReSpeakerArray
        from companion.audio.vad import VoiceActivityDetector

        self._rs = ReSpeakerArray({
            "vendor_id": self.cfg.respeaker.vendor_id,
            "product_id": self.cfg.respeaker.product_id,
        })
        self._ai = AudioInput({
            "sample_rate": self.cfg.audio.sample_rate,
            "channels": self.cfg.audio.channels,
            "chunk_size": self.cfg.audio.chunk_size,
            "input_device_name": self.cfg.audio.input_device_name,
        })
        self._vad = VoiceActivityDetector({"threshold": self.cfg.vad.threshold})
        self._ai.start()
        self._running = True
        self._start_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        import threading
        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self) -> None:
        self._running = False
        self._start_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        if self._ai is not None: self._ai.stop()
        if self._rs is not None: self._rs.stop()

    def _loop(self) -> None:
        while self._running and self._ai is not None:
            c = self._ai.read(timeout=0.3)
            if c is None:
                continue
            rms = float(np.sqrt(np.mean(c**2)))
            doa = float(self._rs.get_doa()) if self._rs is not None else 0.0
            self._vad.process_chunk(c)
            vad_prob = float(getattr(self._vad, "last_prob", 0.0))
            self.chunk_signal.emit(c, rms, doa, vad_prob)

    def _on_chunk(self, chunk, rms, doa, vad_prob) -> None:
        self._wv.push(chunk)
        self._rms.set_rms(min(1.0, rms * 6))
        self._vad_card.set_value(f"{vad_prob:.2f}")
        self._doa.set_angle(doa, active=(rms > 0.05))
        self._doa_label.setText(f"DOA: {doa:4.0f}°  RMS: {rms:.3f}")


# ── LLM tab ─────────────────────────────────────────────────────────────────

class LLMTab(QWidget):
    reply_signal = pyqtSignal(str)
    metric_signal = pyqtSignal(float, float)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._llm = None

        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left); ll.setContentsMargins(12, 12, 12, 12)
        self._model_combo = QComboBox()
        self._model_combo.addItems(list(cfg.llm.model_paths.keys()))
        self._model_combo.setCurrentText(cfg.llm.model)
        self._load_btn = QPushButton("Load")
        self._load_btn.clicked.connect(self._load)
        self._gen_card = MetricCard("Tok/s", "—")
        self._prompt_tok = MetricCard("Prompt eval", "—", "s")
        ll.addWidget(QLabel("Model"))
        ll.addWidget(self._model_combo)
        ll.addWidget(self._load_btn)
        ll.addWidget(self._gen_card)
        ll.addWidget(self._prompt_tok)
        ll.addStretch(1)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right); lr.setContentsMargins(12, 12, 12, 12)
        self._chat = QPlainTextEdit(); self._chat.setReadOnly(True)
        self._input = QLineEdit(); self._input.setPlaceholderText("Ask something…")
        self._input.returnPressed.connect(self._send)
        lr.addWidget(self._chat, 1)
        lr.addWidget(self._input)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0); outer.addWidget(right, 1)

        self.reply_signal.connect(self._append_reply)
        self.metric_signal.connect(lambda t, p: (self._gen_card.set_value(f"{t:.1f}"), self._prompt_tok.set_value(f"{p:.2f}")))

    def _load(self) -> None:
        self._load_btn.setEnabled(False)
        self._load_btn.setText("Loading…")
        self._chat.appendPlainText("[loading model — this can take up to 60 s]")
        import threading

        def _bg():
            from companion.llm.engine import LLMEngine

            self.cfg.llm.model = self._model_combo.currentText()
            try:
                llm = LLMEngine(self.cfg.llm, model_path=self.cfg.llm_model_path())
                llm.load()
                self._llm = llm
                self.reply_signal.emit("[model loaded]")
            except Exception as exc:
                self.reply_signal.emit(f"[load failed: {exc!r}]")
            finally:
                # Re-enable on Qt thread
                self.reply_signal.emit("__LOADED__")

        threading.Thread(target=_bg, daemon=True).start()

    def _send(self) -> None:
        if self._llm is None:
            self._chat.appendPlainText("Load the model first.")
            return
        q = self._input.text().strip()
        if not q:
            return
        self._input.clear()
        self._chat.appendPlainText(f"> {q}")
        import threading
        threading.Thread(target=self._run, args=(q,), daemon=True).start()

    def _run(self, q: str) -> None:
        t0 = time.time()
        out = self._llm.generate(user_message=q, history=[], system_prompt=self.cfg.llm.system_prompt)
        dt = time.time() - t0
        toks = max(1, len(out.split()))
        self.reply_signal.emit(out)
        self.metric_signal.emit(toks / dt, dt)

    def _append_reply(self, text: str) -> None:
        if text == "__LOADED__":
            self._load_btn.setEnabled(True)
            self._load_btn.setText("Load")
            return
        self._chat.appendPlainText(text + "\n")


# ── TTS tab ─────────────────────────────────────────────────────────────────

class TTSTab(QWidget):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._tts = None
        self._out = None

        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left); ll.setContentsMargins(12, 12, 12, 12)
        self._engine = QComboBox(); self._engine.addItems(["kokoro", "piper"])
        self._engine.setCurrentText(cfg.tts.engine)
        self._voice = QLineEdit(cfg.tts.voice)
        self._text = QPlainTextEdit("Hello, I am your companion. How are you feeling today?")
        self._speak_btn = QPushButton("Speak")
        self._speak_btn.clicked.connect(self._speak)
        self._rtf = MetricCard("RTF", "—")
        ll.addWidget(QLabel("Engine")); ll.addWidget(self._engine)
        ll.addWidget(QLabel("Voice")); ll.addWidget(self._voice)
        ll.addWidget(QLabel("Sentence")); ll.addWidget(self._text, 1)
        ll.addWidget(self._speak_btn)
        ll.addWidget(self._rtf)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right); lr.setContentsMargins(12, 12, 12, 12)
        self._status = QLabel("Click Speak to generate audio.")
        lr.addWidget(self._status); lr.addStretch(1)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0); outer.addWidget(right, 1)

    def _speak(self) -> None:
        from companion.audio.io import AudioOutput
        from companion.audio.tts import TextToSpeech

        self.cfg.tts.engine = self._engine.currentText()
        self.cfg.tts.voice = self._voice.text().strip() or "af_heart"
        if self._tts is None or self._tts._active_engine != self.cfg.tts.engine:
            self._tts = TextToSpeech(self.cfg.tts, project_root=self.cfg.project_root)
            self._out = AudioOutput({"output_sample_rate": self._tts.output_sample_rate})
        sentence = self._text.toPlainText().strip()
        import time as _t
        t0 = _t.time()
        pcm = self._tts.synthesize(sentence)
        if pcm is None:
            self._status.setText("Synthesis failed."); return
        duration = len(pcm) / 2 / self._tts.output_sample_rate
        rtf = (_t.time() - t0) / max(0.001, duration)
        self._rtf.set_value(f"{rtf:.2f}")
        self._status.setText(f"{len(pcm) // 2} samples  ·  {duration:.1f} s audio")
        self._out.play_pcm(pcm, self._tts.output_sample_rate)


# ── Vision tab ──────────────────────────────────────────────────────────────

class VisionTab(QWidget):
    state_signal = pyqtSignal(object)

    def __init__(self, cfg: AppConfig, pipeline=None) -> None:
        super().__init__()
        self.cfg = cfg
        self._pipe = pipeline
        self._owns_pipe = pipeline is None

        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left); ll.setContentsMargins(12, 12, 12, 12)
        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start); self._stop_btn.clicked.connect(self._stop)
        self._fps_card = MetricCard("FPS", "—")
        self._lat_card = MetricCard("Latency", "—", "ms")
        ll.addWidget(QLabel("Emotion pipeline"))
        ll.addWidget(self._start_btn); ll.addWidget(self._stop_btn)
        ll.addWidget(self._fps_card); ll.addWidget(self._lat_card)
        ll.addStretch(1)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right); lr.setContentsMargins(12, 12, 12, 12)
        self._circ = EmotionCircumplex()
        self._label = QLabel("—"); self._label.setObjectName("Heading"); self._label.setAlignment(Qt.AlignCenter)
        lr.addWidget(self._label); lr.addWidget(self._circ, 1)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0); outer.addWidget(right, 1)

        self._timer = QTimer(self); self._timer.timeout.connect(self._tick); self._timer.start(50)
        if pipeline is not None:
            self._start_btn.setEnabled(False); self._stop_btn.setEnabled(True)

    def _start(self) -> None:
        if self._pipe is None:
            from companion.vision import EmotionPipeline

            self._pipe = EmotionPipeline({
                "sensor_id": self.cfg.vision.sensor_id, "width": self.cfg.vision.width,
                "height": self.cfg.vision.height, "fps": self.cfg.vision.fps,
                "flip_method": self.cfg.vision.flip_method, "use_csi": self.cfg.vision.use_csi,
                "face_model_path": self.cfg.abspath(self.cfg.vision.face_model_path),
                "emotion_model_path": self.cfg.abspath(self.cfg.vision.emotion_model_path),
                "face_score_threshold": self.cfg.vision.face_score_threshold,
                "smoothing": self.cfg.vision.smoothing,
                "staleness_fade_seconds": self.cfg.vision.staleness_fade_seconds,
            })
            self._pipe.start()
        self._start_btn.setEnabled(False); self._stop_btn.setEnabled(True)

    def _stop(self) -> None:
        if self._owns_pipe and self._pipe is not None:
            self._pipe.stop(); self._pipe = None
        self._start_btn.setEnabled(True); self._stop_btn.setEnabled(False)

    def _tick(self) -> None:
        if self._pipe is None:
            return
        s = self._pipe.get_state()
        self._circ.set_state(s.valence, s.arousal, s.label)
        self._label.setText(f"{s.label}  ·  conf {s.confidence * 100:.0f}%")
        self._fps_card.set_value(f"{s.fps:.1f}")
        self._lat_card.set_value(f"{s.latency_ms:.1f}")


# ── Face tab (pygame preview control) ──────────────────────────────────────

class FaceTab(QWidget):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left); ll.setContentsMargins(12, 12, 12, 12)
        for label, preset in (
            ("Neutral",   (0.0, 0.0)),
            ("Happy",     (+0.7, +0.3)),
            ("Excited",   (+0.7, +0.7)),
            ("Surprised", (+0.1, +0.9)),
            ("Calm",      (+0.5, -0.4)),
            ("Sad",       (-0.6, -0.2)),
            ("Angry",     (-0.7, +0.6)),
            ("Sleep",     (0.0, -0.6)),
        ):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, v=preset, n=label: self._set(v, n))
            ll.addWidget(b)
        self._launch = QPushButton("Launch pygame face window")
        self._launch.clicked.connect(self._launch_face)
        ll.addWidget(self._launch)
        ll.addStretch(1)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right); lr.setContentsMargins(12, 12, 12, 12)
        self._status = QLabel("No face renderer running."); self._status.setAlignment(Qt.AlignCenter)
        lr.addWidget(self._status)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0); outer.addWidget(right, 1)
        self._renderer = None

    def _launch_face(self) -> None:
        from companion.display.renderer import make_renderer

        if self._renderer is not None:
            self._status.setText("Face renderer already running.")
            return
        self._renderer = make_renderer(self.cfg.display)
        if self._renderer is None:
            self._status.setText("No display backend available.")
            return
        self._renderer.set_action_callback(lambda n, p: self._status.setText(f"action: {n}"))
        self._renderer.start()
        self._status.setText("Face running.")

    def _set(self, va: tuple[float, float], name: str) -> None:
        from companion.display.state import FaceState
        if self._renderer is None:
            self._launch_face()
        if self._renderer is None:
            return
        self._renderer.set_face(FaceState(valence=va[0], arousal=va[1]))
        self._status.setText(f"face preset: {name}")


# ── STT tab ─────────────────────────────────────────────────────────────────

class STTTab(QWidget):
    done_signal = pyqtSignal(str, float)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._stt = None
        self._ai = None

        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left); ll.setContentsMargins(12, 12, 12, 12)
        self._backend = QComboBox(); self._backend.addItems(["parakeet", "whisper"])
        self._backend.setCurrentText(cfg.stt.backend)
        self._record_btn = QPushButton("Record 5 s")
        self._record_btn.clicked.connect(self._record)
        self._lat = MetricCard("Latency", "—", "s")
        self._be = MetricCard("Backend", "—")
        ll.addWidget(QLabel("Backend")); ll.addWidget(self._backend)
        ll.addWidget(self._record_btn)
        ll.addWidget(self._lat); ll.addWidget(self._be)
        ll.addStretch(1)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right); lr.setContentsMargins(12, 12, 12, 12)
        self._wv = WaveformView(sample_rate=cfg.audio.sample_rate)
        self._result = QPlainTextEdit(); self._result.setReadOnly(True)
        self._result.setPlaceholderText("Record to see the transcript here.")
        lr.addWidget(self._wv, 1)
        lr.addWidget(self._result, 1)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0); outer.addWidget(right, 1)

        self.done_signal.connect(self._on_done)

    def _record(self) -> None:
        self.cfg.stt.backend = self._backend.currentText()
        self._record_btn.setEnabled(False)
        self._record_btn.setText("Recording…")
        import threading
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from companion.audio.io import AudioInput
        from companion.audio.stt import SpeechToText
        if self._stt is None:
            self._stt = SpeechToText(self.cfg.stt, project_root=self.cfg.project_root)
        ai = AudioInput({
            "sample_rate": self.cfg.audio.sample_rate,
            "channels": self.cfg.audio.channels,
            "chunk_size": self.cfg.audio.chunk_size,
            "input_device_name": self.cfg.audio.input_device_name,
        })
        ai.start()
        chunks = []
        t0 = time.time()
        while time.time() - t0 < 5.0:
            c = ai.read(timeout=0.3)
            if c is not None:
                chunks.append(c)
        ai.stop()
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        t_start = time.time()
        text = self._stt.transcribe(audio)
        self.done_signal.emit(text or "[no speech]", time.time() - t_start)

    def _on_done(self, text: str, latency: float) -> None:
        self._lat.set_value(f"{latency:.2f}")
        self._be.set_value(self._stt.backend if self._stt else "—")
        self._result.setPlainText(text)
        self._record_btn.setEnabled(True)
        self._record_btn.setText("Record 5 s")


# ── VLM tab ─────────────────────────────────────────────────────────────────

class VLMTab(QWidget):
    done_signal = pyqtSignal(str, float)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._vlm = None

        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left); ll.setContentsMargins(12, 12, 12, 12)
        self._q = QLineEdit("What do you see?")
        self._ask = QPushButton("Ask")
        self._ask.clicked.connect(self._run)
        self._caption_btn = QPushButton("Caption current scene")
        self._caption_btn.clicked.connect(self._caption)
        self._lat = MetricCard("Latency", "—", "s")
        ll.addWidget(QLabel("Question"))
        ll.addWidget(self._q)
        ll.addWidget(self._ask)
        ll.addWidget(self._caption_btn)
        ll.addWidget(self._lat)
        ll.addStretch(1)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right); lr.setContentsMargins(12, 12, 12, 12)
        self._result = QPlainTextEdit(); self._result.setReadOnly(True)
        self._result.setPlaceholderText("VLM output will appear here.")
        lr.addWidget(self._result)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0); outer.addWidget(right, 1)
        self.done_signal.connect(self._on_done)

    def _ensure_vlm(self) -> bool:
        if self._vlm is not None:
            return self._vlm.available
        from companion.vision.vlm import MoondreamVLM
        self._vlm = MoondreamVLM(
            self.cfg.abspath(self.cfg.vlm.model_path),
            self.cfg.abspath(self.cfg.vlm.mmproj_path),
            enabled=self.cfg.vlm.enabled,
            max_tokens=self.cfg.vlm.max_tokens,
        )
        return self._vlm.available

    def _grab_frame(self):
        from companion.vision.camera import CSICamera
        cam = CSICamera(
            sensor_id=self.cfg.vision.sensor_id, width=self.cfg.vision.width,
            height=self.cfg.vision.height, fps=self.cfg.vision.fps,
            flip_method=self.cfg.vision.flip_method, use_csi=self.cfg.vision.use_csi,
        )
        frame = None
        for _ in range(40):
            frame = cam.read()
            if frame is not None:
                break
            time.sleep(0.1)
        cam.close()
        return frame

    def _run(self) -> None:
        import threading
        self._ask.setEnabled(False)
        self._result.setPlainText("Thinking…")
        q = self._q.text().strip() or "What do you see?"
        threading.Thread(target=self._bg, args=(q,), daemon=True).start()

    def _caption(self) -> None:
        import threading
        self._caption_btn.setEnabled(False)
        self._result.setPlainText("Captioning…")
        threading.Thread(target=self._bg, args=("__CAPTION__",), daemon=True).start()

    def _bg(self, q: str) -> None:
        if not self._ensure_vlm():
            self.done_signal.emit("[VLM unavailable — check model files]", 0.0)
            return
        frame = self._grab_frame()
        if frame is None:
            self.done_signal.emit("[no frame from camera]", 0.0)
            return
        t0 = time.time()
        out = self._vlm.caption(frame) if q == "__CAPTION__" else self._vlm.answer(frame, q)
        self.done_signal.emit(out or "[no output]", time.time() - t0)

    def _on_done(self, text: str, latency: float) -> None:
        self._result.setPlainText(text)
        self._lat.set_value(f"{latency:.2f}")
        self._ask.setEnabled(True)
        self._caption_btn.setEnabled(True)


# ── Tools tab ───────────────────────────────────────────────────────────────

class ToolsTab(QWidget):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._fg = None

        left = QFrame(); left.setObjectName("Card")
        ll = QVBoxLayout(left); ll.setContentsMargins(12, 12, 12, 12)
        self._q = QLineEdit("set a timer for 5 minutes")
        self._detect = QPushButton("Detect")
        self._detect.clicked.connect(self._detect_tool)
        ll.addWidget(QLabel("User utterance"))
        ll.addWidget(self._q)
        ll.addWidget(self._detect)
        ll.addWidget(QLabel("Registered tools:"))
        self._tool_list = QPlainTextEdit(); self._tool_list.setReadOnly(True)
        ll.addWidget(self._tool_list, 1)

        right = QFrame(); right.setObjectName("Card")
        lr = QVBoxLayout(right); lr.setContentsMargins(12, 12, 12, 12)
        self._result = QPlainTextEdit(); self._result.setReadOnly(True)
        lr.addWidget(self._result)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left, 0); outer.addWidget(right, 1)

        from companion.tools import registry
        registry.load_all_tools()
        names = "\n".join(f" • {s['name']}  —  {s['description']}" for s in registry.all_schemas())
        self._tool_list.setPlainText(names)

    def _detect_tool(self) -> None:
        from companion.llm.function_gemma import FunctionGemma
        from companion.tools import registry
        if self._fg is None:
            self._fg = FunctionGemma(
                self.cfg.abspath(self.cfg.function_gemma.model_path),
                enabled=self.cfg.function_gemma.enabled,
                confidence_threshold=self.cfg.function_gemma.confidence_threshold,
            )
            if self._fg.available:
                self._fg.set_tools(registry.all_schemas())
        if not self._fg.available:
            self._result.setPlainText("FunctionGemma unavailable — run scripts/download_models.py.")
            return
        call = self._fg.detect(self._q.text())
        if call is None:
            self._result.setPlainText("No tool call detected.")
            return
        out = registry.invoke(call.name, call.args)
        self._result.setPlainText(
            f"Tool   : {call.name}\nArgs   : {call.args}\nResult : {out}"
        )


# ── Main window ─────────────────────────────────────────────────────────────

class DebugGUI(QMainWindow):
    def __init__(self, cfg: AppConfig, conversation=None, emotion=None, respeaker=None, scene=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Companion · Debug")
        self.resize(1200, 780)
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._tabs.addTab(AudioTab(cfg), "Audio")
        self._tabs.addTab(STTTab(cfg), "STT")
        self._tabs.addTab(LLMTab(cfg), "LLM")
        self._tabs.addTab(TTSTab(cfg), "TTS")
        self._tabs.addTab(VisionTab(cfg, pipeline=emotion), "Vision")
        self._tabs.addTab(VLMTab(cfg), "VLM")
        self._tabs.addTab(ToolsTab(cfg), "Tools")
        self._tabs.addTab(FaceTab(cfg), "Face")


def launch(cfg=None, conversation=None, emotion=None, respeaker=None, scene=None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    if cfg is None:
        cfg = load_config(os.path.join(_ROOT, "config.yaml"))
    w = DebugGUI(cfg, conversation, emotion, respeaker, scene)
    w.show()
    app.exec_()


def main() -> int:
    cfg = load_config(os.path.join(_ROOT, "config.yaml"))
    setup_logging(cfg.app.log_level)
    launch(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
