"""Debug GUI — single PyQt5 window with a tab per subsystem.

One launch point for every subsystem test. Tabs follow the conversation
data flow: Audio in → STT → LLM → Tools → Memory → TTS → Vision → VLM →
Face display → Face tracking.

Each tab uses the same layout contract:
  • left card: Heading, subtitle, controls, bottom-anchored status line
  • right card: the subsystem's live output (preview, metrics, transcript…)

Launch standalone:
    python3 -m tests.debug_gui

Or from the main app's View → Debug GUI menu.
"""

from __future__ import annotations

import copy as _copy
import logging
import os
import sys
import math
import threading
import time
from typing import Optional

import cv2  # noqa: F401 — imported first so we can strip its Qt plugin override
import numpy as np

# opencv-contrib-python ships its own Qt5 plugins and sets
# QT_QPA_PLATFORM_PLUGIN_PATH on import. That path doesn't contain a
# working xcb plugin on Jetson, so PyQt5 fails to start with:
#   "Could not load the Qt platform plugin 'xcb' in .../cv2/qt/plugins"
# Strip the override here so PyQt5 uses its own plugins at QApplication
# creation time.
if "cv2/qt/plugins" in os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtCore import Qt, QTimer, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QImage, QPixmap  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
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


# ── Shared helpers ──────────────────────────────────────────────────────────

_BGR_TO_PIXMAP_DIAG_LOGGED = False


def _bgr_to_pixmap(frame: Optional[np.ndarray], max_w: int = 480) -> QPixmap:
    """Convert numpy BGR (or BGRA / gray) frame → QPixmap, scaled to max_w.

    Defensive against:
      - cameras that return 4-channel (BGRA / BGRx) frames
      - non-contiguous arrays (causes garbled / all-one-colour output)
      - short-lived numpy buffers (forces a deep copy into QImage storage)

    Logs one diagnostic line the first time it runs so we can tell whether
    the camera is actually outputting green vs. a Qt-side mishap.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return QPixmap()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        return QPixmap()

    global _BGR_TO_PIXMAP_DIAG_LOGGED
    if not _BGR_TO_PIXMAP_DIAG_LOGGED:
        try:
            b, g, r = frame.mean(axis=(0, 1))
            log.warning(
                f"[debug_gui] first frame: shape={frame.shape} dtype={frame.dtype} "
                f"BGR means=({b:.0f}, {g:.0f}, {r:.0f})"
            )
        except Exception:
            pass
        _BGR_TO_PIXMAP_DIAG_LOGGED = True

    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(max_w * h / w)))
        h, w = frame.shape[:2]

    # BGR→RGB via channel reverse + explicit copy guarantees a fresh,
    # contiguous buffer even if cv2.cvtColor returned a view.
    rgb = np.ascontiguousarray(frame[..., ::-1])
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    # .copy() forces QImage to own its pixel data so rgb can be GC'd safely.
    return QPixmap.fromImage(qimg.copy())


_PREVIEW_STYLE = (
    f"background: {PALETTE['bg']};"
    f" border: 1px dashed {PALETTE['surface2']};"
    f" border-radius: 6px;"
    f" color: {PALETTE['text_dim']};"
)


def _preview_label(placeholder: str, min_h: int = 260) -> QLabel:
    lbl = QLabel(placeholder)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setMinimumHeight(min_h)
    lbl.setStyleSheet(_PREVIEW_STYLE)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {PALETTE['surface2']};")
    return f


def _btn_row(*buttons: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    for b in buttons:
        row.addWidget(b)
    return row


def _persist_doa_offset(config_path: str, offset_deg: float) -> None:
    """Write respeaker.doa_offset_deg back to config.yaml without disturbing
    comments or unrelated keys.

    The match captures three groups:
      \\g<1>  leading indent + "doa_offset_deg:" + space after colon
      \\g<2>  the value (everything up to whitespace-before-'#' or line end)
      \\g<3>  any trailing whitespace + inline comment, preserved verbatim
    We substitute only \\g<2>. Critically, \\g<3> is kept as-is so the
    leading whitespace before the '#' stays intact — otherwise YAML
    parses `-89.90# comment` as a single string and the next boot crashes.
    """
    import re
    with open(config_path) as fh:
        text = fh.read()
    pattern = re.compile(
        r"(^\s*doa_offset_deg:\s*)(\S+)(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    replacement = rf"\g<1>{offset_deg:.2f}\g<3>"
    new_text, n = pattern.subn(replacement, text, count=1)
    if n == 0:
        # Key missing — append it under the respeaker block.
        new_text = re.sub(
            r"(^respeaker:\n(?:\s+\S.*\n)+)",
            rf"\1  doa_offset_deg: {offset_deg:.2f}\n",
            text, count=1, flags=re.MULTILINE,
        )
    with open(config_path, "w") as fh:
        fh.write(new_text)


class SharedHead:
    """Lightweight reference to the currently-running HeadController.

    Set by FaceTrackingTab on start, cleared on stop. Other tabs (Audio
    calibration) read it to correlate DOA with the head's commanded pan.
    """

    def __init__(self) -> None:
        self._head = None
        self._lock = threading.Lock()

    def set(self, head) -> None:
        with self._lock:
            self._head = head

    def clear(self) -> None:
        with self._lock:
            self._head = None

    @property
    def head(self):
        return self._head


class SharedRenderer:
    """Lightweight reference to the currently-running face renderer.

    Set by FaceTab on launch, cleared on stop. Vision tab reads it to
    mirror the detected emotion onto the face display in real time.
    """

    def __init__(self) -> None:
        self._renderer = None
        self._lock = threading.Lock()

    def set(self, renderer) -> None:
        with self._lock:
            self._renderer = renderer

    def clear(self) -> None:
        with self._lock:
            self._renderer = None

    @property
    def renderer(self):
        return self._renderer


# Map EmotionClassifier labels → FaceState.expression hints so detected
# emotions get the same distinctive ornaments as the Face-display presets.
_EMOTION_TO_EXPRESSION = {
    "Happiness": "excited",
    "Surprise":  "surprised",
    "Fear":      "confused",
    "Sadness":   "sad",
    "Anger":     "angry",
    "Disgust":   "angry",
    # Contempt / Neutral have no distinctive ornament → None (pure V/A).
}


class SharedVision:
    """Reference-counted shared EmotionPipeline.

    Both the Vision tab and the Face-tracking tab need a running
    EmotionPipeline, and `/dev/video0` can only be opened once. This
    wrapper starts the pipeline on the first acquire and stops it when
    the last consumer releases — allowing both tabs to run concurrently
    against a single camera + detector + classifier thread.

    If `preexisting` is given (e.g. the main app already has a pipeline
    running), acquire/release return that pipeline but never stop it.
    """

    def __init__(self, cfg: AppConfig, preexisting=None) -> None:
        self.cfg = cfg
        self._pipe = preexisting
        self._external = preexisting is not None
        self._refs = 0
        self._lock = threading.Lock()

    def acquire(self):
        """Start the pipeline if needed; return the running instance."""
        with self._lock:
            if self._pipe is None:
                from companion.vision import EmotionPipeline
                vcfg = (self.cfg.vision.__dict__
                        if hasattr(self.cfg.vision, "__dict__")
                        else dict(self.cfg.vision))
                self._pipe = EmotionPipeline(vcfg)
                self._pipe.start()
                time.sleep(0.4)  # warm-up for camera + detector
            self._refs += 1
            return self._pipe

    def release(self) -> None:
        with self._lock:
            if self._refs > 0:
                self._refs -= 1
            if self._refs == 0 and self._pipe is not None and not self._external:
                try:
                    self._pipe.stop()
                except Exception as exc:
                    log.debug(f"SharedVision stop failed: {exc!r}")
                self._pipe = None

    @property
    def pipe(self):
        """Current pipeline (may be None if not acquired)."""
        return self._pipe


class _Scaffold:
    """Two-column tab layout: left card (controls + status), right (output).

    Usage:
        s = _Scaffold(self, "Audio + DOA", "Mic waveform, VAD, ReSpeaker DOA.")
        s.ll.addWidget(...)           # controls
        s.lr.addWidget(...)           # live output
        s.finalize()                  # pins status to bottom of left
        # self._status is s.status — set text via self.set_status(...)
    """

    def __init__(self, widget: QWidget, title: str, subtitle: str = "") -> None:
        self.widget = widget
        self.left = QFrame(); self.left.setObjectName("Card")
        self.right = QFrame(); self.right.setObjectName("Card")

        self.ll = QVBoxLayout(self.left)
        self.ll.setContentsMargins(14, 14, 14, 14)
        self.ll.setSpacing(8)
        self.lr = QVBoxLayout(self.right)
        self.lr.setContentsMargins(14, 14, 14, 14)
        self.lr.setSpacing(8)

        heading = QLabel(title); heading.setObjectName("Heading")
        self.ll.addWidget(heading)
        if subtitle:
            sub = QLabel(subtitle); sub.setObjectName("Subtle")
            sub.setWordWrap(True)
            self.ll.addWidget(sub)
        self.ll.addSpacing(4)

        self.status = QLabel("Idle.")
        self.status.setObjectName("Subtle")
        self.status.setWordWrap(True)

        outer = QHBoxLayout(widget)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        outer.addWidget(self.left, 0)
        outer.addWidget(self.right, 1)
        self.left.setMinimumWidth(270)
        self.left.setMaximumWidth(320)

    def finalize(self) -> None:
        self.ll.addStretch(1)
        self.ll.addWidget(_divider())
        self.ll.addWidget(self.status)




# ── Audio tab ───────────────────────────────────────────────────────────────

class AudioTab(QWidget):
    chunk_signal = pyqtSignal(object, float, float, float)  # chunk, rms, signed_doa, vad
    # Cross-thread finaliser for DOA calibration. We need a real signal
    # here (not QTimer.singleShot) because the worker runs outside the
    # Qt event loop and a static-timer callback posted from there never
    # fires. offset=NaN is the sentinel for "no offset to apply".
    cal_done_signal = pyqtSignal(float, str)

    def __init__(self, cfg: AppConfig, shared_head: Optional["SharedHead"] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._shared_head = shared_head
        self._ai = None
        self._rs = None
        self._vad = None
        self._running = False
        self._cal_busy = False

        s = _Scaffold(self, "Audio + DOA",
                      "Live mic waveform, VAD, and ReSpeaker direction of arrival "
                      "(signed, body-frame: 0° = front, + = right, − = left). "
                      "The yellow marker on the polar plot shows the head's current pan.")

        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        s.ll.addLayout(_btn_row(self._start_btn, self._stop_btn))

        self._vad_card = MetricCard("VAD prob", "0.00")
        self._rms = RMSBar()
        metrics = QHBoxLayout(); metrics.setSpacing(8)
        metrics.addWidget(self._vad_card); metrics.addWidget(self._rms)
        s.ll.addLayout(metrics)

        # Calibration controls ───────────────────────────────────────────
        s.ll.addWidget(_divider())
        cal_heading = QLabel("DOA calibration"); cal_heading.setObjectName("Subtle")
        s.ll.addWidget(cal_heading)
        self._offset_card = MetricCard("Offset", f"{cfg.respeaker.doa_offset_deg:.1f}", "°")
        s.ll.addWidget(self._offset_card)
        self._zero_btn = QPushButton("Zero here")
        self._zero_btn.setToolTip(
            "Stand directly in front, speak for ~2 s, then click. "
            "Captures current DOA as the new body-forward zero."
        )
        self._auto_btn = QPushButton("From head")
        self._auto_btn.setToolTip(
            "Requires Face tracking to be running with a locked face. "
            "Samples (DOA − head_pan) for 2 s and saves as offset."
        )
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Set offset back to 0°.")
        self._zero_btn.clicked.connect(self._cal_zero)
        self._auto_btn.clicked.connect(self._cal_auto)
        self._reset_btn.clicked.connect(self._cal_reset)
        s.ll.addLayout(_btn_row(self._zero_btn, self._auto_btn, self._reset_btn))
        for b in (self._zero_btn, self._auto_btn, self._reset_btn):
            b.setEnabled(False)

        self._wv = WaveformView(sample_rate=cfg.audio.sample_rate)
        self._doa = DOAPolarPlot()
        self._doa_label = QLabel("DOA: —   ·   head: —   ·   RMS: 0.000")
        self._doa_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s.lr.addWidget(self._wv, 1)
        s.lr.addWidget(self._doa_label)
        s.lr.addWidget(self._doa, 1)

        self._scaffold = s
        self.chunk_signal.connect(self._on_chunk)
        self.cal_done_signal.connect(self._on_cal_done)
        # Poll head pan at 10 Hz so the yellow marker follows the head.
        self._head_timer = QTimer(self); self._head_timer.timeout.connect(self._refresh_head_marker)
        self._head_timer.start(100)
        s.finalize()

    def set_status(self, text: str) -> None:
        self._scaffold.status.setText(text)

    def _start(self) -> None:
        from companion.audio.io import AudioInput
        from companion.audio.respeaker import ReSpeakerArray
        from companion.audio.vad import VoiceActivityDetector

        # ReSpeaker is optional — missing device shouldn't kill the tab.
        try:
            rs = ReSpeakerArray({
                "vendor_id": self.cfg.respeaker.vendor_id,
                "product_id": self.cfg.respeaker.product_id,
                "doa_offset_deg": self.cfg.respeaker.doa_offset_deg,
            })
            self._rs = rs if getattr(rs, "is_connected", False) else None
        except Exception as exc:
            log.warning(f"ReSpeaker init failed: {exc!r}")
            self._rs = None

        try:
            self._ai = AudioInput({
                "sample_rate": self.cfg.audio.sample_rate,
                "channels": self.cfg.audio.channels,
                "chunk_size": self.cfg.audio.chunk_size,
                "input_device_name": self.cfg.audio.input_device_name,
            })
            self._vad = VoiceActivityDetector({"threshold": self.cfg.vad.threshold})
            self._ai.start()
        except Exception as exc:
            # Inline cleanup so we can preserve the error message in the status.
            err = f"Start failed: {exc}"
            if self._ai is not None:
                try: self._ai.stop()
                except Exception: pass
            if self._rs is not None:
                try: self._rs.stop()
                except Exception: pass
            self._ai = self._rs = self._vad = None
            self._running = False
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self.set_status(err)
            return

        self._running = True
        self._start_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        for b in (self._zero_btn, self._auto_btn, self._reset_btn):
            b.setEnabled(self._rs is not None)
        self.set_status(
            f"Running.  ReSpeaker: {'connected' if self._rs else 'not found (DOA disabled)'}"
        )
        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self) -> None:
        self._running = False
        self._start_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        for b in (self._zero_btn, self._auto_btn, self._reset_btn):
            b.setEnabled(False)
        if self._ai is not None:
            try: self._ai.stop()
            except Exception: pass
        if self._rs is not None:
            try: self._rs.stop()
            except Exception: pass
        self._ai = self._rs = self._vad = None
        self.set_status("Stopped.")

    def _loop(self) -> None:
        raw_log_next = time.time() + 1.0
        last_raw: Optional[float] = None
        raw_stuck_counter = 0
        while self._running:
            # Snapshot refs each iteration — _stop() nulls these out from the
            # GUI thread and we must not AttributeError mid-read.
            ai = self._ai
            rs = self._rs
            vad = self._vad
            if ai is None:
                return
            c = ai.read(timeout=0.3)
            if c is None:
                continue
            rms = float(np.sqrt(np.mean(c**2)))
            raw_doa = float("nan")
            try:
                if rs is not None:
                    raw_doa = float(rs.get_doa())
                    doa = float(rs.get_doa_signed())
                else:
                    doa = float("nan")
            except Exception:
                doa = float("nan")
            # Diagnostic: log raw DOA once per second and detect a frozen
            # chip (same value for N consecutive reads when the RMS is high
            # enough to plausibly be speech).
            if not np.isnan(raw_doa):
                if last_raw is not None and abs(raw_doa - last_raw) < 0.5 and rms > 0.05:
                    raw_stuck_counter += 1
                else:
                    raw_stuck_counter = 0
                last_raw = raw_doa
                now = time.time()
                if now >= raw_log_next:
                    stuck_note = (
                        f" [stuck {raw_stuck_counter} frames — speak louder / clap]"
                        if raw_stuck_counter > 30 else ""
                    )
                    log.info(f"DOA raw={raw_doa:5.1f}° signed={doa:+6.1f}° "
                             f"rms={rms:.3f}{stuck_note}")
                    raw_log_next = now + 1.0
            if vad is not None:
                try: vad.process_chunk(c)
                except Exception: pass
            vad_prob = float(getattr(vad, "last_prob", 0.0)) if vad is not None else 0.0
            self.chunk_signal.emit(c, rms, doa, vad_prob)

    # ── UI update slots ─────────────────────────────────────────────────
    def _current_head_pan(self) -> Optional[float]:
        if self._shared_head is None or self._shared_head.head is None:
            return None
        try:
            pan, _tilt = self._shared_head.head.get_head_pose()
            return float(pan)
        except Exception:
            return None

    def _refresh_head_marker(self) -> None:
        pan = self._current_head_pan()
        self._doa.set_head_pan(pan)

    def _on_chunk(self, chunk, rms, doa, vad_prob) -> None:
        self._wv.push(chunk)
        self._rms.set_rms(min(1.0, rms * 6))
        self._vad_card.set_value(f"{vad_prob:.2f}")
        head_pan = self._current_head_pan()
        head_str = f"{head_pan:+5.1f}°" if head_pan is not None else "—"
        raw_str = "—"
        if self._rs is not None:
            try:
                raw_str = f"{int(self._rs.get_doa()):3d}°"
            except Exception:
                pass
        # Gate readings on RMS: the ReSpeaker holds its last angle in
        # silence, so anything below ~0.05 RMS is almost always stale.
        # When stale we dim the label and the polar marker and append
        # "(stale)" so the user isn't misled by a held direction.
        live = rms > 0.05
        live_color = PALETTE["text"]
        stale_color = PALETTE["muted"]
        if np.isnan(doa):
            self._doa.set_angle(0.0, active=False)
            self._doa_label.setStyleSheet(f"color: {stale_color};")
            self._doa_label.setText(
                f"raw: {raw_str}   ·   body: —   ·   head: {head_str}   ·   RMS: {rms:.3f}"
            )
        else:
            self._doa.set_angle(doa, active=live)
            self._doa_label.setStyleSheet(f"color: {live_color if live else stale_color};")
            tag = "" if live else "   ·   (stale)"
            self._doa_label.setText(
                f"raw: {raw_str}   ·   body: {doa:+6.1f}°   ·   "
                f"head: {head_str}   ·   RMS: {rms:.3f}{tag}"
            )

    # ── Calibration actions ─────────────────────────────────────────────
    def _sample_raw_doa(self, duration_s: float = 2.0) -> Optional[float]:
        """Average the raw (unsigned 0-359) DOA over `duration_s`.
        Uses a circular-mean so wraparound near 360 doesn't smear.

        Per-sample exceptions are swallowed (USB hiccups happen) — we
        only return None if the entire window produced zero samples.
        Also logs timing so we can tell whether get_doa() is just slow
        (blocking HID transfer) vs. raising.
        """
        if self._rs is None:
            return None
        import math
        xs, ys = 0.0, 0.0
        n = 0
        errs = 0
        slowest = 0.0
        t0 = time.time()
        while time.time() - t0 < duration_s:
            t_call = time.time()
            try:
                raw = float(self._rs.get_doa())
            except Exception as exc:
                errs += 1
                if errs == 1:
                    log.warning(f"get_doa() raised: {exc!r} (suppressing further)")
                time.sleep(0.05)
                continue
            dt = time.time() - t_call
            slowest = max(slowest, dt)
            xs += math.cos(math.radians(raw))
            ys += math.sin(math.radians(raw))
            n += 1
            # Only sleep if get_doa was fast; otherwise it already ate the budget.
            if dt < 0.05:
                time.sleep(0.05 - dt)
        log.info(
            f"DOA sample window: elapsed={time.time()-t0:.2f}s samples={n} "
            f"errs={errs} slowest_call={slowest*1000:.0f}ms"
        )
        if n == 0:
            return None
        mean = math.degrees(math.atan2(ys / n, xs / n))
        if mean < 0:
            mean += 360.0
        return mean

    def _apply_offset(self, new_offset: float) -> None:
        # Normalise to (-180, 180]
        off = new_offset % 360.0
        if off > 180.0:
            off -= 360.0
        self.cfg.respeaker.doa_offset_deg = float(off)
        if self._rs is not None:
            self._rs.doa_offset_deg = float(off)
        self._offset_card.set_value(f"{off:.1f}")
        # Flush stale trail so the plot only reflects post-calibration direction.
        self._doa.clear_trail()
        try:
            _persist_doa_offset(
                os.path.join(self.cfg.project_root, "config.yaml"),
                off,
            )
            self.set_status(f"Offset set to {off:+.1f}° and saved to config.yaml.")
        except Exception as exc:
            self.set_status(f"Offset set to {off:+.1f}° (save failed: {exc!r}).")

    def _start_cal(self, label: str) -> None:
        self._cal_busy = True
        for b in (self._zero_btn, self._auto_btn, self._reset_btn):
            b.setEnabled(False)
        self.set_status(f"{label} — sampling 2 s…")
        # Watchdog — even if the worker thread or Qt slot gets lost,
        # force-recover after 10 s. Audio-input and ReSpeaker HID share
        # the same USB device, and a blocking `get_doa()` call can spike
        # to several seconds on a busy bus, so 5 s was too tight.
        QTimer.singleShot(10000, self._cal_watchdog)

    def _cal_watchdog(self) -> None:
        if self._cal_busy:
            log.warning("Calibration watchdog fired — worker never completed.")
            self.set_status("Calibration timed out (check logs). Buttons re-enabled.")
            self._end_cal()

    def _end_cal(self) -> None:
        self._cal_busy = False
        for b in (self._zero_btn, self._auto_btn, self._reset_btn):
            b.setEnabled(self._rs is not None)

    def _on_cal_done(self, offset: float, msg: str) -> None:
        """Qt-thread slot invoked via `cal_done_signal.emit(...)` from the
        calibration worker. offset=NaN means 'no offset to apply, just
        unstick the tab and show the message'. try/except/finally here is
        the last line of defence — _end_cal() MUST run.
        """
        try:
            if not math.isnan(offset):
                self._apply_offset(offset)
            if msg:
                self.set_status(msg)
        except Exception as exc:
            log.exception("DOA calibration apply failed")
            self.set_status(f"Calibration error: {exc}")
        finally:
            self._end_cal()

    def _cal_zero(self) -> None:
        if self._rs is None or self._cal_busy:
            return
        self._start_cal("Zero here")

        def _bg():
            # Any failure in the worker still has to signal the main
            # thread — otherwise the tab stays stuck in sampling state
            # until the watchdog fires.
            try:
                raw = self._sample_raw_doa(2.0)
                if raw is None:
                    self.cal_done_signal.emit(float("nan"),
                                              "Calibration failed — no DOA samples.")
                    return
                self.cal_done_signal.emit(float(raw), "")
            except Exception as exc:
                log.exception("Zero-calibration worker crashed")
                self.cal_done_signal.emit(float("nan"),
                                          f"Calibration crashed: {exc}")
        threading.Thread(target=_bg, daemon=True).start()

    def _cal_auto(self) -> None:
        if self._rs is None or self._cal_busy:
            return
        if self._current_head_pan() is None:
            self.set_status("Start Face tracking first (needed for head-pose ground truth).")
            return
        self._start_cal("Auto-calibrate from head")

        def _bg():
            try:
                raw = self._sample_raw_doa(2.0)
                pan = self._current_head_pan()
                if raw is None or pan is None:
                    self.cal_done_signal.emit(float("nan"),
                                              "Calibration failed — missing DOA or head pose.")
                    return
                self.cal_done_signal.emit(float(raw - pan), "")
            except Exception as exc:
                log.exception("Auto-calibration worker crashed")
                self.cal_done_signal.emit(float("nan"),
                                          f"Calibration crashed: {exc}")
        threading.Thread(target=_bg, daemon=True).start()

    def _cal_reset(self) -> None:
        # Reset is a safety valve — if an earlier calibration left
        # _cal_busy stuck True, force it false here before applying so the
        # tab can always be recovered without restarting.
        self._cal_busy = False
        try:
            self._apply_offset(0.0)
        except Exception as exc:
            log.exception("DOA reset failed")
            self.set_status(f"Reset error: {exc}")
        self._end_cal()


# ── STT tab ─────────────────────────────────────────────────────────────────

class STTTab(QWidget):
    done_signal = pyqtSignal(str, float)
    chunk_signal = pyqtSignal(object)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._stt = None

        s = _Scaffold(self, "Speech-to-Text",
                      "Record 5 seconds, transcribe via Parakeet (or Whisper fallback). "
                      "Waveform updates live during recording.")

        self._backend = QComboBox(); self._backend.addItems(["parakeet", "whisper"])
        self._backend.setCurrentText(cfg.stt.backend)
        self._record_btn = QPushButton("Record 5 s")
        self._record_btn.clicked.connect(self._record)
        self._lat = MetricCard("Latency", "—", "s")
        self._be = MetricCard("Backend", cfg.stt.backend)
        s.ll.addWidget(QLabel("Backend")); s.ll.addWidget(self._backend)
        s.ll.addWidget(self._record_btn)
        metrics = QHBoxLayout(); metrics.setSpacing(8)
        metrics.addWidget(self._lat); metrics.addWidget(self._be)
        s.ll.addLayout(metrics)

        self._wv = WaveformView(sample_rate=cfg.audio.sample_rate)
        self._result = QPlainTextEdit(); self._result.setReadOnly(True)
        self._result.setPlaceholderText("Transcript will appear here.")
        s.lr.addWidget(self._wv, 1)
        s.lr.addWidget(self._result, 1)

        self._scaffold = s
        self.chunk_signal.connect(self._wv.push)
        self.done_signal.connect(self._on_done)
        s.finalize()

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    def _record(self) -> None:
        self.cfg.stt.backend = self._backend.currentText()
        self._record_btn.setEnabled(False)
        self._record_btn.setText("Recording…")
        self.set_status("Recording 5 s…")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from companion.audio.io import AudioInput
        from companion.audio.stt import SpeechToText
        try:
            if self._stt is None:
                self._stt = SpeechToText(self.cfg.stt, project_root=self.cfg.project_root)
            ai = AudioInput({
                "sample_rate": self.cfg.audio.sample_rate,
                "channels": self.cfg.audio.channels,
                "chunk_size": self.cfg.audio.chunk_size,
                "input_device_name": self.cfg.audio.input_device_name,
            })
            ai.start()
        except Exception as exc:
            self.done_signal.emit(f"[record failed: {exc!r}]", 0.0)
            return

        chunks: list = []
        try:
            t0 = time.time()
            while time.time() - t0 < 5.0:
                c = ai.read(timeout=0.3)
                if c is not None:
                    chunks.append(c)
                    self.chunk_signal.emit(c)
        finally:
            try: ai.stop()
            except Exception: pass

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        # Diagnostics — ALSA can silently drop frames and leave us with zero
        # chunks while PortAudio prints only warnings. Log what we got so
        # "[no speech]" failures have a paper trail.
        n = audio.size
        peak = float(np.max(np.abs(audio))) if n else 0.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if n else 0.0
        log.info(
            f"STT record: chunks={len(chunks)} samples={n} "
            f"({n/16000:.2f}s) peak={peak:.3f} rms={rms:.4f}"
        )
        t_start = time.time()
        try:
            text = self._stt.transcribe(audio)
        except Exception as exc:
            text = f"[transcribe failed: {exc!r}]"
        # Give the user an informative hint instead of silent "[no speech]"
        # when the stream produced no real audio.
        if not text:
            if n == 0:
                text = "[no audio captured — ALSA/PortAudio didn't deliver any frames]"
            elif peak < 0.01:
                text = f"[no speech — mic was silent (peak={peak:.4f}, rms={rms:.5f})]"
            else:
                text = "[no speech detected in audio]"
        self.done_signal.emit(text, time.time() - t_start)

    def _on_done(self, text: str, latency: float) -> None:
        self._lat.set_value(f"{latency:.2f}")
        if self._stt is not None:
            self._be.set_value(self._stt.backend)
        self._result.setPlainText(text)
        self._record_btn.setEnabled(True)
        self._record_btn.setText("Record 5 s")
        self.set_status(f"Done · {latency:.2f} s.")


# ── LLM tab ─────────────────────────────────────────────────────────────────

class LLMTab(QWidget):
    reply_signal = pyqtSignal(str)
    metric_signal = pyqtSignal(float, float)
    loaded_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._llm = None
        self._generating = False

        s = _Scaffold(self, "LLM chat",
                      "Load a Gemma 4 variant and chat. Tok/s and prompt-eval time update per turn.")

        self._model_combo = QComboBox()
        self._model_combo.addItems(list(cfg.llm.model_paths.keys()))
        self._model_combo.setCurrentText(cfg.llm.model)
        self._load_btn = QPushButton("Load model")
        self._load_btn.clicked.connect(self._load)
        self._gen_card = MetricCard("Tok/s", "—")
        self._prompt_tok = MetricCard("Prompt eval", "—", "s")
        s.ll.addWidget(QLabel("Model")); s.ll.addWidget(self._model_combo)
        s.ll.addWidget(self._load_btn)
        metrics = QHBoxLayout(); metrics.setSpacing(8)
        metrics.addWidget(self._gen_card); metrics.addWidget(self._prompt_tok)
        s.ll.addLayout(metrics)

        self._chat = QPlainTextEdit(); self._chat.setReadOnly(True)
        self._chat.setPlaceholderText("Load a model first, then type below.")
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask something and press Enter…")
        self._input.returnPressed.connect(self._send)
        self._input.setEnabled(False)
        s.lr.addWidget(self._chat, 1)
        s.lr.addWidget(self._input)

        self._scaffold = s
        self.reply_signal.connect(self._append_reply)
        self.metric_signal.connect(
            lambda t, p: (self._gen_card.set_value(f"{t:.1f}"),
                          self._prompt_tok.set_value(f"{p:.2f}"))
        )
        self.loaded_signal.connect(self._on_loaded)
        s.finalize()

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    def _load(self) -> None:
        self.cfg.llm.model = self._model_combo.currentText()
        try:
            path = self.cfg.llm_model_path()
        except Exception as exc:
            self._chat.appendPlainText(f"[bad model key: {exc}]")
            self.set_status("Bad model key.")
            return
        if not os.path.exists(path):
            self._chat.appendPlainText(
                f"[model file missing: {path}\n run scripts/download_models.py]"
            )
            self.set_status("Model file missing.")
            return

        self._load_btn.setEnabled(False)
        self._load_btn.setText("Loading…")
        self.set_status("Loading model — this can take up to 60 s.")
        self._chat.appendPlainText(f"[loading {self.cfg.llm.model}…]")

        def _bg():
            import traceback
            try:
                from companion.llm.engine import LLMEngine
                llm = LLMEngine(self.cfg.llm, model_path=path)
                llm.load()
                self._llm = llm
                self.loaded_signal.emit(True, f"Loaded {self.cfg.llm.model}.")
            except Exception as exc:
                log.error("LLM load failed:\n%s", traceback.format_exc())
                self.loaded_signal.emit(False, f"Load failed: {exc}")

        threading.Thread(target=_bg, daemon=True).start()

    def _on_loaded(self, ok: bool, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._load_btn.setText("Load model")
        self._input.setEnabled(ok)
        self._chat.appendPlainText(f"[{msg}]\n")
        self.set_status(msg)
        if ok:
            self._input.setFocus()

    def _send(self) -> None:
        if self._llm is None:
            self._chat.appendPlainText("[load a model first]")
            self.set_status("Click Load model first.")
            return
        if self._generating:
            return
        q = self._input.text().strip()
        if not q:
            return
        self._input.clear()
        self._input.setEnabled(False)
        self._generating = True
        self._chat.appendPlainText(f"> {q}")
        self.set_status("Generating…")
        threading.Thread(target=self._run, args=(q,), daemon=True).start()

    def _run(self, q: str) -> None:
        # Whole body under try — any crash MUST emit reply_signal so the
        # _generating flag gets cleared in _append_reply and the input
        # re-enables. Otherwise the user can't send a follow-up message.
        try:
            t0 = time.time()
            try:
                out = self._llm.generate(
                    user_message=q, history=[],
                    system_prompt=self.cfg.llm.system_prompt,
                )
            except Exception as exc:
                self.reply_signal.emit(f"[generation failed: {exc!r}]")
                return
            dt = max(1e-6, time.time() - t0)
            text = out if isinstance(out, str) and out else "[empty response]"
            toks = max(1, len(text.split()))
            self.reply_signal.emit(text)
            self.metric_signal.emit(toks / dt, dt)
        except Exception as exc:
            log.exception("LLM worker crashed")
            self.reply_signal.emit(f"[LLM crashed: {exc!r}]")

    def _append_reply(self, text: str) -> None:
        self._chat.appendPlainText(text + "\n")
        self._generating = False
        self._input.setEnabled(True)
        self._input.setFocus()
        self.set_status("Ready.")


# ── TTS tab ─────────────────────────────────────────────────────────────────

class TTSTab(QWidget):
    done_signal = pyqtSignal(bool, str, str, float)  # ok, status, preview, rtf

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._tts = None
        self._out = None
        self._speaking = False

        s = _Scaffold(self, "Text-to-Speech",
                      "Synthesise a sentence with Kokoro (natural) or Piper (fast). "
                      "RTF <1.0 means faster-than-realtime. "
                      "⚠ Stop Audio / STT first — simultaneous capture + playback "
                      "crashes PortAudio on Jetson ALSA.")

        self._engine = QComboBox(); self._engine.addItems(["kokoro", "piper"])
        self._engine.setCurrentText(cfg.tts.engine)
        self._voice = QLineEdit(cfg.tts.voice)
        self._text = QPlainTextEdit("Hello, I am your companion. How are you feeling today?")
        self._text.setMaximumHeight(120)
        self._speak_btn = QPushButton("Speak")
        self._speak_btn.clicked.connect(self._speak)
        self._rtf = MetricCard("RTF", "—")
        s.ll.addWidget(QLabel("Engine")); s.ll.addWidget(self._engine)
        s.ll.addWidget(QLabel("Voice")); s.ll.addWidget(self._voice)
        s.ll.addWidget(QLabel("Sentence")); s.ll.addWidget(self._text, 1)
        s.ll.addWidget(self._speak_btn)
        s.ll.addWidget(self._rtf)

        self._preview = _preview_label("Click Speak to synthesise.")
        s.lr.addWidget(self._preview, 1)

        self._scaffold = s
        self.done_signal.connect(self._on_done)
        s.finalize()

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    def _speak(self) -> None:
        if self._speaking:
            return
        self.cfg.tts.engine = self._engine.currentText()
        self.cfg.tts.voice = self._voice.text().strip() or "af_heart"
        sentence = self._text.toPlainText().strip()
        if not sentence:
            self.set_status("Enter a sentence first.")
            return

        self._speaking = True
        self._speak_btn.setEnabled(False)
        self.set_status("Synthesising…")
        threading.Thread(target=self._bg, args=(sentence,), daemon=True).start()

    def _bg(self, sentence: str) -> None:
        """Runs on a worker thread. Touches no Qt widgets; emits done_signal.

        Whole body is inside an outer try so a completely unexpected
        crash (module import failure, math error on edge-case audio) still
        emits done_signal and re-enables the Speak button.
        """
        try:
            from companion.audio.io import AudioOutput
            from companion.audio.tts import TextToSpeech

            try:
                if self._tts is None or self._tts.active_engine != self.cfg.tts.engine:
                    self._tts = TextToSpeech(self.cfg.tts, project_root=self.cfg.project_root)
                    self._out = AudioOutput({"output_sample_rate": self._tts.output_sample_rate})
            except Exception as exc:
                self.done_signal.emit(False, f"TTS init failed: {exc!r}", "", 0.0)
                return

            t0 = time.time()
            try:
                pcm = self._tts.synthesize(sentence)
            except Exception as exc:
                self.done_signal.emit(False, f"Synthesis failed: {exc!r}", "", 0.0)
                return
            if pcm is None:
                self.done_signal.emit(False, "Synthesis failed (no audio).", "", 0.0)
                return

            duration = len(pcm) / 2 / self._tts.output_sample_rate
            rtf = (time.time() - t0) / max(0.001, duration)
            preview = (
                f"{len(pcm) // 2} samples  ·  {duration:.1f} s of audio\n\n"
                f"Engine: {self._tts.active_engine}   Voice: {self.cfg.tts.voice}"
            )
            try:
                self._out.play_pcm(pcm, self._tts.output_sample_rate)
            except Exception as exc:
                self.done_signal.emit(False, f"Playback failed: {exc!r}", preview, rtf)
                return
            self.done_signal.emit(True, f"Played · RTF {rtf:.2f}.", preview, rtf)
        except Exception as exc:
            log.exception("TTS worker crashed")
            self.done_signal.emit(False, f"TTS crashed: {exc!r}", "", 0.0)

    def _on_done(self, ok: bool, status: str, preview: str, rtf: float) -> None:
        self._speaking = False
        self._speak_btn.setEnabled(True)
        if rtf > 0:
            self._rtf.set_value(f"{rtf:.2f}")
        if preview:
            self._preview.setText(preview)
        self.set_status(status)


# ── Tools tab ───────────────────────────────────────────────────────────────

class ToolsTab(QWidget):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._fg = None

        s = _Scaffold(self, "Tool routing",
                      "FunctionGemma-270M parses an utterance into a structured tool call.")

        self._q = QLineEdit("set a timer for 5 minutes")
        self._detect = QPushButton("Detect")
        self._detect.clicked.connect(self._detect_tool)
        s.ll.addWidget(QLabel("User utterance"))
        s.ll.addWidget(self._q)
        s.ll.addWidget(self._detect)
        s.ll.addWidget(QLabel("Registered tools:"))
        self._tool_list = QPlainTextEdit(); self._tool_list.setReadOnly(True)
        s.ll.addWidget(self._tool_list, 1)

        self._result = QPlainTextEdit(); self._result.setReadOnly(True)
        self._result.setPlaceholderText("Parsed tool call + result will appear here.")
        s.lr.addWidget(self._result)

        try:
            from companion.tools import registry
            registry.load_all_tools()
            names = "\n".join(
                f" • {t['name']}  —  {t['description']}" for t in registry.all_schemas()
            )
            self._tool_list.setPlainText(names or "(no tools registered)")
        except Exception as exc:
            self._tool_list.setPlainText(f"[tool registry failed: {exc!r}]")

        self._scaffold = s
        s.finalize()

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    def _detect_tool(self) -> None:
        from companion.llm.function_gemma import FunctionGemma
        from companion.tools import registry

        self._detect.setEnabled(False)
        self.set_status("Detecting…")
        try:
            if self._fg is None:
                self._fg = FunctionGemma(
                    self.cfg.abspath(self.cfg.function_gemma.model_path),
                    enabled=self.cfg.function_gemma.enabled,
                    confidence_threshold=self.cfg.function_gemma.confidence_threshold,
                )
                if self._fg.available:
                    self._fg.set_tools(registry.all_schemas())
            if not self._fg.available:
                self._result.setPlainText(
                    "FunctionGemma unavailable — run scripts/download_models.py."
                )
                self.set_status("Model unavailable.")
                return
            call = self._fg.detect(self._q.text())
            if call is None:
                self._result.setPlainText("No tool call detected.")
                self.set_status("No match.")
                return
            out = registry.invoke(call.name, call.args)
            self._result.setPlainText(
                f"Tool   : {call.name}\nArgs   : {call.args}\nResult : {out}"
            )
            self.set_status(f"Matched: {call.name}")
        except Exception as exc:
            self._result.setPlainText(f"[tool detection failed: {exc!r}]")
            self.set_status("Failed.")
        finally:
            self._detect.setEnabled(True)


# ── Memory tab ──────────────────────────────────────────────────────────────

class MemoryTab(QWidget):
    # action: "add" | "search" | "forget" | "error"
    done_signal = pyqtSignal(str, str, list)
    status_signal = pyqtSignal(str)  # used from worker threads

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._mem = None
        self._mem_lock = threading.Lock()

        s = _Scaffold(self, "Memory",
                      "Speaker-scoped facts stored in Mem0 + ChromaDB. "
                      "Add, search, or forget per speaker.")

        self._speaker = QLineEdit("Yogee")
        self._text = QPlainTextEdit("User likes chai and reads on Tuesday evenings.")
        self._text.setMaximumHeight(110)
        self._query = QLineEdit("what does user like")
        self._add_btn = QPushButton("Add")
        self._search_btn = QPushButton("Search")
        self._forget_btn = QPushButton("Forget")
        self._forget_btn.setToolTip("Delete all memories for this speaker")
        self._add_btn.clicked.connect(self._add)
        self._search_btn.clicked.connect(self._search)
        self._forget_btn.clicked.connect(self._forget)
        s.ll.addWidget(QLabel("Speaker")); s.ll.addWidget(self._speaker)
        s.ll.addWidget(QLabel("Memory to add")); s.ll.addWidget(self._text, 1)
        s.ll.addWidget(QLabel("Search query")); s.ll.addWidget(self._query)
        s.ll.addLayout(_btn_row(self._add_btn, self._search_btn, self._forget_btn))

        self._results_header = QLabel("Search results:")
        self._results = QListWidget()
        self._results.setStyleSheet(
            f"QListWidget::item {{ padding: 6px 4px; border-bottom: 1px solid {PALETTE['surface2']}; }}"
        )
        s.lr.addWidget(self._results_header)
        s.lr.addWidget(self._results, 1)

        self._scaffold = s
        self.done_signal.connect(self._on_done)
        self.status_signal.connect(self._scaffold.status.setText)
        s.finalize()

    def set_status(self, t: str) -> None:
        """Thread-safe: emits via signal so QLabel is only touched on the GUI thread."""
        self.status_signal.emit(t)

    def _ensure(self) -> bool:
        """Construct the MemoryStore lazily. Safe under concurrent workers."""
        with self._mem_lock:
            if self._mem is not None:
                return self._mem.available
            self.set_status("Initialising memory store…")
            try:
                from companion.llm.memory import MemoryStore
                self._mem = MemoryStore(
                    self.cfg.abspath(self.cfg.memory.chroma_dir),
                    enabled=self.cfg.memory.enabled,
                    top_k=self.cfg.memory.top_k,
                )
            except Exception as exc:
                self.set_status(f"Memory init failed: {exc!r}")
                return False
            if self._mem.available:
                self.set_status("Memory ready.")
                return True
            self.set_status("Memory unavailable (mem0ai / chromadb).")
            return False

    def _busy(self, on: bool) -> None:
        for b in (self._add_btn, self._search_btn, self._forget_btn):
            b.setEnabled(not on)

    def _add(self) -> None:
        text = self._text.toPlainText().strip()
        sp = self._speaker.text().strip() or "unknown"
        if not text:
            self.set_status("Nothing to add — type a memory first.")
            return
        self._busy(True); self.set_status("Adding…")

        def _bg():
            # Outer try so even a crash in _ensure() still emits done_signal
            # and the buttons come back — otherwise tab stays busy forever.
            try:
                if not self._ensure():
                    self.done_signal.emit("error", "Memory unavailable.", [])
                    return
                self._mem.add(text, sp)
                self.done_signal.emit("add", f"Added for '{sp}'.", [])
            except Exception as exc:
                log.exception("Memory add crashed")
                self.done_signal.emit("error", f"Add failed: {exc!r}", [])
        threading.Thread(target=_bg, daemon=True).start()

    def _search(self) -> None:
        q = self._query.text().strip()
        sp = self._speaker.text().strip() or "unknown"
        if not q:
            self.set_status("Type a search query first.")
            return
        self._busy(True); self.set_status("Searching…")

        def _bg():
            try:
                if not self._ensure():
                    self.done_signal.emit("error", "Memory unavailable.", [])
                    return
                hits = self._mem.retrieve(q, sp)
                self.done_signal.emit(
                    "search", f"{len(hits)} hit(s) for '{sp}'.", list(hits),
                )
            except Exception as exc:
                log.exception("Memory search crashed")
                self.done_signal.emit("error", f"Search failed: {exc!r}", [])
        threading.Thread(target=_bg, daemon=True).start()

    def _forget(self) -> None:
        sp = self._speaker.text().strip() or "unknown"
        self._busy(True); self.set_status(f"Forgetting all for '{sp}'…")

        def _bg():
            try:
                if not self._ensure():
                    self.done_signal.emit("error", "Memory unavailable.", [])
                    return
                self._mem.forget(sp)
                self.done_signal.emit("forget", f"Cleared all memories for '{sp}'.", [])
            except Exception as exc:
                log.exception("Memory forget crashed")
                self.done_signal.emit("error", f"Forget failed: {exc!r}", [])
        threading.Thread(target=_bg, daemon=True).start()

    def _on_done(self, action: str, msg: str, hits: list) -> None:
        self.set_status(msg)
        self._busy(False)
        if action == "search":
            self._results.clear()
            if hits:
                for h in hits:
                    self._results.addItem(str(h))
            else:
                self._results.addItem("(no hits)")
        elif action == "forget":
            self._results.clear()
        # "add" and "error" leave existing search results alone.


# ── Vision tab ──────────────────────────────────────────────────────────────

class VisionTab(QWidget):
    ready_signal = pyqtSignal(bool, str)  # ok, status message

    def __init__(
        self,
        cfg: AppConfig,
        shared_vision: Optional["SharedVision"] = None,
        pipeline=None,
        shared_renderer: Optional["SharedRenderer"] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # Normalise to a SharedVision so the lifecycle logic is uniform.
        if shared_vision is not None:
            self._sv = shared_vision
        elif pipeline is not None:
            self._sv = SharedVision(cfg, preexisting=pipeline)
        else:
            self._sv = SharedVision(cfg)
        self._shared_renderer = shared_renderer
        self._acquired = False
        self._last_mirrored_label: Optional[str] = None

        s = _Scaffold(self, "Vision + emotion",
                      "Live camera, face bbox, YOLO pose detector, and the emotion circumplex. "
                      "Can run simultaneously with Face tracking (they share one camera + pipeline). "
                      "Tick 'Mirror to face' to stream the detected emotion to the face display.")

        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        self._fps_card = MetricCard("FPS", "—")
        self._lat_card = MetricCard("Latency", "—", "ms")
        s.ll.addLayout(_btn_row(self._start_btn, self._stop_btn))
        metrics = QHBoxLayout(); metrics.setSpacing(8)
        metrics.addWidget(self._fps_card); metrics.addWidget(self._lat_card)
        s.ll.addLayout(metrics)

        self._mirror = QCheckBox("Mirror to face display")
        self._mirror.setToolTip(
            "Stream the detected valence / arousal / expression to the "
            "face display. Launch 'Face display' first — otherwise this "
            "just shows a warning."
        )
        s.ll.addWidget(self._mirror)

        self._preview = _preview_label("Click Start — camera preview appears here.", min_h=260)
        self._label = QLabel("—"); self._label.setObjectName("Heading")
        self._label.setAlignment(Qt.AlignCenter)
        self._circ = EmotionCircumplex()
        s.lr.addWidget(self._preview, 2)
        s.lr.addWidget(self._label)
        s.lr.addWidget(self._circ, 2)

        self._scaffold = s
        s.finalize()

        self._timer = QTimer(self); self._timer.timeout.connect(self._tick)
        self.ready_signal.connect(self._on_ready)
        # Auto-start if wrapping an externally-owned live pipeline.
        if pipeline is not None or (shared_vision is not None and shared_vision.pipe is not None):
            self._acquired = True
            self._sv.acquire()
            self._start_btn.setEnabled(False); self._stop_btn.setEnabled(True)
            self._timer.start(66)

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    def _start(self) -> None:
        try:
            pipe = self._sv.acquire()
            self._acquired = True
        except Exception as exc:
            self.set_status(f"Start failed: {exc}")
            return
        backend = getattr(pipe.camera, "backend", "?")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)   # Stop remains clickable during the wait
        self.set_status(f"Waiting for first frame (camera: {backend})…")

        def _wait_for_frame():
            # Outer try so a crash inside pipe.get_state() still gets the
            # tab out of "Waiting for first frame…" state.
            try:
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    if not self._acquired:
                        return  # user clicked Stop
                    if pipe.get_state().frame is not None:
                        self.ready_signal.emit(True, f"Running · camera: {backend}")
                        return
                    time.sleep(0.05)
                self.ready_signal.emit(
                    False,
                    f"Camera opened ({backend}) but no frames in 3 s — "
                    "check /dev/video0 / CSI cable / use_csi in config.",
                )
            except Exception as exc:
                log.exception("Vision wait-for-frame crashed")
                self.ready_signal.emit(False, f"Start crashed: {exc}")
        threading.Thread(target=_wait_for_frame, daemon=True).start()

    def _on_ready(self, ok: bool, msg: str) -> None:
        if not self._acquired:
            return  # user already cancelled
        self.set_status(msg)
        if ok:
            self._timer.start(66)
        else:
            # Frame timeout — release and reset UI to idle.
            self._sv.release()
            self._acquired = False
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    def _stop(self) -> None:
        self._timer.stop()
        if self._acquired:
            self._sv.release()
            self._acquired = False
        self._start_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._preview.setText("Stopped.")
        self._preview.setPixmap(QPixmap())
        self.set_status("Stopped.")

    def _tick(self) -> None:
        pipe = self._sv.pipe
        if pipe is None:
            return
        s = pipe.get_state()
        self._circ.set_state(s.valence, s.arousal, s.label)
        self._label.setText(f"{s.label}  ·  conf {s.confidence * 100:.0f}%")
        self._fps_card.set_value(f"{s.fps:.1f}")
        self._lat_card.set_value(f"{s.latency_ms:.1f}")
        if s.frame is not None:
            frame = s.frame.copy()
            if s.bbox is not None:
                x, y, w, h = s.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 120), 2)
                cv2.putText(frame, f"{s.label} {s.confidence*100:.0f}%",
                            (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 120), 1, cv2.LINE_AA)
            self._preview.setPixmap(_bgr_to_pixmap(frame, max_w=520))

        if self._mirror.isChecked():
            self._mirror_to_face(s)

    def _mirror_to_face(self, s) -> None:
        """Forward the detected emotion to the face renderer.

        No-op if no face display has been launched; keeps the detection
        free-running either way. Status line tells the user what's happening.
        """
        from companion.display.state import FaceState
        renderer = (
            self._shared_renderer.renderer
            if self._shared_renderer is not None else None
        )
        if renderer is None:
            # Don't spam the status line every 66 ms — only change on transitions.
            if self._last_mirrored_label != "__no_renderer__":
                self.set_status(
                    "Mirror on, but no face display running — launch Face display."
                )
                self._last_mirrored_label = "__no_renderer__"
            return
        expression = _EMOTION_TO_EXPRESSION.get(s.label) if s.has_face else None
        try:
            renderer.set_face(FaceState(
                valence=float(s.valence),
                arousal=float(s.arousal),
                expression=expression,
            ))
        except Exception as exc:
            log.debug(f"mirror_to_face failed: {exc!r}")
            return
        if s.label != self._last_mirrored_label:
            self.set_status(f"Mirroring '{s.label}' to face display.")
            self._last_mirrored_label = s.label


# ── VLM tab ─────────────────────────────────────────────────────────────────

class VLMTab(QWidget):
    done_signal = pyqtSignal(str, float)
    frame_signal = pyqtSignal(object)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._vlm = None

        s = _Scaffold(self, "Scene understanding (VLM)",
                      "Captures one frame from the camera and asks Moondream-2. "
                      "Stop Vision / Face tracking first if the camera is busy.")

        self._q = QLineEdit("What do you see?")
        self._ask = QPushButton("Ask")
        self._ask.clicked.connect(self._run)
        self._caption_btn = QPushButton("Caption scene")
        self._caption_btn.clicked.connect(self._caption)
        self._lat = MetricCard("Latency", "—", "s")
        s.ll.addWidget(QLabel("Question"))
        s.ll.addWidget(self._q)
        s.ll.addLayout(_btn_row(self._ask, self._caption_btn))
        s.ll.addWidget(self._lat)

        self._preview = _preview_label("No frame captured yet.", min_h=220)
        self._result = QPlainTextEdit(); self._result.setReadOnly(True)
        self._result.setPlaceholderText("VLM output will appear here.")
        s.lr.addWidget(self._preview, 3)
        s.lr.addWidget(self._result, 2)

        self._scaffold = s
        self.done_signal.connect(self._on_done)
        self.frame_signal.connect(
            lambda f: self._preview.setPixmap(_bgr_to_pixmap(f, max_w=520))
        )
        s.finalize()

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

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
        try:
            cam = CSICamera(
                sensor_id=self.cfg.vision.sensor_id, width=self.cfg.vision.width,
                height=self.cfg.vision.height, fps=self.cfg.vision.fps,
                flip_method=self.cfg.vision.flip_method, use_csi=self.cfg.vision.use_csi,
            )
        except Exception as exc:
            log.warning(f"Camera open failed: {exc!r}")
            return None
        frame = None
        for _ in range(40):
            frame = cam.read()
            if frame is not None:
                break
            time.sleep(0.1)
        cam.close()
        return frame

    def _set_busy(self, on: bool) -> None:
        self._ask.setEnabled(not on)
        self._caption_btn.setEnabled(not on)

    def _run(self) -> None:
        self._set_busy(True)
        self.set_status("Capturing frame…")
        self._result.setPlainText("Thinking…")
        q = self._q.text().strip() or "What do you see?"
        threading.Thread(target=self._bg, args=(q,), daemon=True).start()

    def _caption(self) -> None:
        self._set_busy(True)
        self.set_status("Capturing frame…")
        self._result.setPlainText("Captioning…")
        threading.Thread(target=self._bg, args=("__CAPTION__",), daemon=True).start()

    def _bg(self, q: str) -> None:
        # Outer try — _ensure_vlm or _grab_frame can raise on first call
        # (model loading, camera init); a crash here must still emit
        # done_signal or the Ask / Caption buttons stay disabled forever.
        try:
            if not self._ensure_vlm():
                self.done_signal.emit("[VLM unavailable — check model files]", 0.0)
                return
            frame = self._grab_frame()
            if frame is None:
                self.done_signal.emit(
                    "[no frame from camera — is another tab holding it?]", 0.0,
                )
                return
            self.frame_signal.emit(frame)
            t0 = time.time()
            try:
                out = self._vlm.caption(frame) if q == "__CAPTION__" else self._vlm.answer(frame, q)
            except Exception as exc:
                out = f"[inference failed: {exc!r}]"
            self.done_signal.emit(out or "[no output]", time.time() - t0)
        except Exception as exc:
            log.exception("VLM worker crashed")
            self.done_signal.emit(f"[VLM crashed: {exc!r}]", 0.0)

    def _on_done(self, text: str, latency: float) -> None:
        self._result.setPlainText(text)
        self._lat.set_value(f"{latency:.2f}")
        self._set_busy(False)
        if latency > 0:
            self.set_status(f"Done · {latency:.2f} s.")
        else:
            self.set_status("Failed.")


# ── Face display tab ────────────────────────────────────────────────────────

class FaceTab(QWidget):
    # (valence, arousal, sleep, expression_hint). Each preset picks a V/A
    # pair AND an explicit expression hint so the renderer draws
    # distinctive ornaments (swirl eyes for confused, lightbulb for idea,
    # heart eyes for love, etc.) instead of blurring together.
    PRESETS = (
        ("Neutral",   (0.0,   0.0, False, None)),
        ("Excited",   (+0.8, +0.9, False, "excited")),
        ("Surprised", (+0.1, +0.95, False, "surprised")),
        ("Confused",  (-0.25, +0.35, False, "confused")),
        ("Thinking",  (+0.05, -0.1, False, "thinking")),
        ("Idea",      (+0.7, +0.6, False, "idea")),
        ("Wink",      (+0.5, +0.3, False, "wink")),
        ("Listening", (+0.1, +0.1, False, "listening")),
        ("Sad",       (-0.7, -0.3, False, "sad")),
        ("Angry",     (-0.7, +0.7, False, "angry")),
        ("Sleep",     (0.0,  -0.8, True,  None)),
    )

    def __init__(
        self,
        cfg: AppConfig,
        shared_renderer: Optional["SharedRenderer"] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._shared_renderer = shared_renderer
        self._renderer = None

        s = _Scaffold(self, "Face display",
                      "Drive the ESP32 touchscreen (or HDMI pygame fallback). "
                      "Pick a preset; the current emotion is shown on the right. "
                      "Launch once to let Vision mirror detected emotions here.")

        self._launch_btn = QPushButton("Launch face")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._launch_btn.clicked.connect(self._launch_face)
        self._stop_btn.clicked.connect(self._stop)
        s.ll.addLayout(_btn_row(self._launch_btn, self._stop_btn))

        s.ll.addWidget(QLabel("Presets:"))
        grid_outer = QVBoxLayout(); grid_outer.setSpacing(6)
        current_row: Optional[QHBoxLayout] = None
        for i, (name, params) in enumerate(self.PRESETS):
            if i % 2 == 0:
                current_row = QHBoxLayout(); current_row.setSpacing(6)
                grid_outer.addLayout(current_row)
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, p=params, n=name: self._set(p, n))
            current_row.addWidget(b)
        s.ll.addLayout(grid_outer)

        self._current = QLabel("No face renderer running.")
        self._current.setObjectName("Heading")
        self._current.setAlignment(Qt.AlignCenter)
        self._info = _preview_label("Backend: (not started)\nValence / arousal shown here.", min_h=180)
        s.lr.addWidget(self._current)
        s.lr.addWidget(self._info, 1)

        self._scaffold = s
        s.finalize()

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    def _launch_face(self) -> None:
        from companion.display.renderer import make_renderer
        if self._renderer is not None:
            self.set_status("Already running.")
            return
        try:
            self._renderer = make_renderer(self.cfg.display)
        except Exception as exc:
            self._renderer = None
            self.set_status(f"Renderer init failed: {exc!r}")
            return
        if self._renderer is None:
            self.set_status("No display backend available.")
            return
        self._renderer.set_action_callback(
            lambda n, p: self._current.setText(f"action: {n}")
        )
        self._renderer.start()
        if self._shared_renderer is not None:
            self._shared_renderer.set(self._renderer)
        self._info.setText(f"Backend: {type(self._renderer).__name__}\nNo preset selected yet.")
        self._launch_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.set_status("Face running.")
        self._current.setText("Neutral")

    def _stop(self) -> None:
        if self._shared_renderer is not None:
            self._shared_renderer.clear()
        if self._renderer is not None:
            try: self._renderer.stop()
            except Exception: pass
        self._renderer = None
        self._launch_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._current.setText("No face renderer running.")
        self._info.setText("Backend: (not started)\nValence / arousal shown here.")
        self.set_status("Stopped.")

    def _set(self, params: tuple, name: str) -> None:
        from companion.display.state import FaceState
        if self._renderer is None:
            self._launch_face()
        if self._renderer is None:
            return
        v, a, sleep, expression = params
        self._renderer.set_face(FaceState(
            valence=v, arousal=a, sleep=sleep, expression=expression,
        ))
        self._current.setText(name)
        flags = []
        if sleep:
            flags.append("sleep")
        if expression:
            flags.append(f"expr:{expression}")
        flag_str = f"  ({', '.join(flags)})" if flags else ""
        self._info.setText(
            f"Backend: {type(self._renderer).__name__}\n"
            f"Preset: {name}{flag_str}\nValence: {v:+.2f}   Arousal: {a:+.2f}"
        )
        self.set_status(f"Preset: {name}")


# ── Face tracking tab ───────────────────────────────────────────────────────

class FaceTrackingTab(QWidget):
    ready_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        cfg: AppConfig,
        shared_vision: Optional["SharedVision"] = None,
        shared_head: Optional["SharedHead"] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._sv = shared_vision if shared_vision is not None else SharedVision(cfg)
        self._shared_head = shared_head
        self._acquired = False
        self._head = None
        self._tracker = None

        s = _Scaffold(self, "Face tracking",
                      "Live YOLO-pose camera + proportional servo control. Shares the "
                      "EmotionPipeline with the Vision tab — both can run concurrently. "
                      "Hover any slider for a tuning hint; see the panel below for full guide.")

        self._sim = QCheckBox("Sim motors (safe)")
        self._sim.setChecked(True)
        self._sim.setToolTip(
            "Sim mode: servos aren't driven, but the tracker still runs and "
            "the preview shows where it would move. Use this before switching "
            "to REAL motors to confirm framing and sensitivity."
        )

        _KP_HELP = (
            "kp — proportional gain. How hard the head chases the face error each tick.\n"
            "  • Low (0.05–0.15): very smooth but laggy — head trails the face.\n"
            "  • Mid (0.20–0.40): good general balance. Start here.\n"
            "  • High (0.60+): snappy but can overshoot and oscillate.\n"
            "Tune up until the head tracks fast moves, then back off until oscillation stops."
        )
        _DEADBAND_HELP = (
            "deadband — dead zone, in degrees. Errors smaller than this don't move the head.\n"
            "  • Small (0–2°): very twitchy — servos micro-adjust on detection noise.\n"
            "  • Mid (3–6°): stays still when the face is centred, chases real motion.\n"
            "  • Large (10°+): face can drift noticeably before the head reacts.\n"
            "Increase if the head jitters while looking straight at you; decrease if it feels sluggish."
        )

        self._kp_slider = QSlider(Qt.Horizontal)
        self._kp_slider.setRange(5, 150)                 # 0.05 .. 1.50, step 0.01
        self._kp_slider.setValue(30)                     # default 0.30
        self._kp_slider.setToolTip(_KP_HELP)
        self._kp_value_lbl = QLabel("0.30")
        self._kp_value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._kp_slider.valueChanged.connect(
            lambda v: self._kp_value_lbl.setText(f"{v/100:.2f}")
        )
        self._deadband_slider = QSlider(Qt.Horizontal)
        self._deadband_slider.setRange(0, 150)           # 0.0 .. 15.0, step 0.1
        self._deadband_slider.setValue(40)               # default 4.0°
        self._deadband_slider.setToolTip(_DEADBAND_HELP)
        self._deadband_value_lbl = QLabel("4.0°")
        self._deadband_value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._deadband_slider.valueChanged.connect(
            lambda v: self._deadband_value_lbl.setText(f"{v/10:.1f}°")
        )
        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        self._fps_card = MetricCard("FPS", "—")

        s.ll.addWidget(self._sim)

        def _slider_row(name: str, slider: QSlider, value_lbl: QLabel,
                        hint: str = "") -> QVBoxLayout:
            col = QVBoxLayout(); col.setSpacing(2)
            top = QHBoxLayout(); top.setSpacing(6)
            top.addWidget(QLabel(name)); top.addStretch(1); top.addWidget(value_lbl)
            col.addLayout(top); col.addWidget(slider)
            if hint:
                h = QLabel(hint); h.setObjectName("Subtle")
                h.setWordWrap(True)
                col.addWidget(h)
            return col

        s.ll.addLayout(_slider_row(
            "kp", self._kp_slider, self._kp_value_lbl,
            hint="how hard it chases — start 0.30; higher = snappier, lower = smoother",
        ))
        s.ll.addLayout(_slider_row(
            "deadband", self._deadband_slider, self._deadband_value_lbl,
            hint="dead zone — raise if it jitters at rest, lower if it feels sluggish",
        ))
        s.ll.addLayout(_btn_row(self._start_btn, self._stop_btn))
        s.ll.addWidget(self._fps_card)

        # Preview + tuning reference panel
        self._preview = _preview_label("Live camera + pose overlay appears here when started.", min_h=320)
        s.lr.addWidget(self._preview, 1)

        tuning_hdr = QLabel("Tuning guide")
        tuning_hdr.setObjectName("Heading")
        s.lr.addWidget(tuning_hdr)
        tuning = QLabel(
            "<b>kp (proportional gain)</b> — how hard the head chases the face each tick.<br>"
            "&nbsp;&nbsp;• 0.05–0.15: smooth / laggy &nbsp;·&nbsp; "
            "0.20–0.40: balanced (start here) &nbsp;·&nbsp; "
            "0.60+: snappy, may oscillate.<br>"
            "&nbsp;&nbsp;<i>Tune up until it tracks fast motion, then back off until oscillation stops.</i>"
            "<br><br>"
            "<b>deadband (°)</b> — errors smaller than this don't move the head.<br>"
            "&nbsp;&nbsp;• 0–2°: twitchy (reacts to detection noise) &nbsp;·&nbsp; "
            "3–6°: balanced &nbsp;·&nbsp; 10°+: face drifts before reaction.<br>"
            "&nbsp;&nbsp;<i>Raise if the head jitters when you look straight at the camera; "
            "lower if it feels sluggish.</i>"
            "<br><br>"
            "<b>Sim motors</b> — rehearse without torque. The preview still shows where the "
            "head would go. Switch to REAL only after the preview looks right."
            "<br><br>"
            "<b>Not tunable here</b> (edit config.yaml if needed): camera HFOV 62° / VFOV 37°, "
            "control rate 15 Hz, soft limits from motor calibration."
        )
        tuning.setWordWrap(True)
        tuning.setTextFormat(Qt.RichText)
        tuning.setObjectName("Subtle")
        s.lr.addWidget(tuning)

        self._scaffold = s
        s.finalize()

        self._timer = QTimer(self); self._timer.timeout.connect(self._tick)
        self.ready_signal.connect(self._on_ready)

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    def _kp_value(self) -> float:
        return self._kp_slider.value() / 100.0

    def _deadband_value(self) -> float:
        return self._deadband_slider.value() / 10.0

    def _set_config_enabled(self, enabled: bool) -> None:
        self._sim.setEnabled(enabled)
        self._kp_slider.setEnabled(enabled)
        self._deadband_slider.setEnabled(enabled)

    def _start(self) -> None:
        try:
            pipe = self._sv.acquire()
            self._acquired = True
        except Exception as exc:
            self.set_status(f"Start failed: {exc}")
            return
        backend = getattr(pipe.camera, "backend", "?")
        self.set_status(f"Waiting for first frame (camera: {backend})…")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._set_config_enabled(False)

        # Off-thread wait + motor setup so the GUI stays responsive.
        # Cleanup on any failure happens here; _stop() is the user-initiated
        # path and should not run mid-start.
        sim = self._sim.isChecked()
        kp = self._kp_value()
        db = self._deadband_value()

        def _bg():
            from companion.motor.controller import HeadController
            from companion.vision.face_tracker import FaceTracker

            t0 = time.time()
            while time.time() - t0 < 3.0:
                if not self._acquired:
                    return  # user cancelled
                if pipe.get_state().frame is not None:
                    break
                time.sleep(0.05)
            else:
                self.ready_signal.emit(
                    False,
                    f"Camera opened ({backend}) but no frames in 3 s — "
                    "check /dev/video0 / CSI cable / use_csi in config.",
                )
                return

            try:
                motor_cfg = _copy.deepcopy(self.cfg.motor)
                motor_cfg.sim_only = sim
                head = HeadController(motor_cfg)
                head.connect()
                head.enable_torque(True)
            except Exception as exc:
                self.ready_signal.emit(False, f"Motor init failed: {exc}")
                return

            try:
                tracker = FaceTracker(
                    head=head, vision=pipe,
                    kp=kp, deadband_deg=db,
                    update_hz=15.0,
                    camera_hfov_deg=62.0, camera_vfov_deg=37.0,
                )
                tracker.start_async()
            except Exception as exc:
                try: head.disconnect()
                except Exception: pass
                self.ready_signal.emit(False, f"Tracker start failed: {exc}")
                return

            self._head = head
            self._tracker = tracker
            if self._shared_head is not None:
                self._shared_head.set(head)

            mode = "SIM" if sim else "REAL"
            self.ready_signal.emit(
                True,
                f"Running ({mode} motors) · cam: {backend} · "
                f"kp={kp:.2f} · db={db:.1f}°",
            )
        threading.Thread(target=_bg, daemon=True).start()

    def _on_ready(self, ok: bool, msg: str) -> None:
        if not self._acquired:
            return
        self.set_status(msg)
        if ok:
            self._timer.start(66)
        else:
            # Failed — unwind what we acquired.
            self._sv.release()
            self._acquired = False
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._set_config_enabled(True)

    def _stop(self) -> None:
        self._timer.stop()
        if self._shared_head is not None:
            self._shared_head.clear()
        if self._tracker is not None:
            try: self._tracker.stop()
            except Exception: pass
        if self._head is not None:
            try: self._head.disconnect()
            except Exception: pass
        self._tracker = self._head = None
        if self._acquired:
            self._sv.release()
            self._acquired = False
        self._start_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._set_config_enabled(True)
        self._preview.setText("Stopped.")
        self._preview.setPixmap(QPixmap())
        self.set_status("Stopped.")

    def _tick(self) -> None:
        if self._tracker is None:
            return
        from companion.vision.face_tracker import render_annotated_frame
        snap = self._tracker.latest_snapshot()
        img = render_annotated_frame(snap) if snap is not None else None
        pipe = self._sv.pipe
        if img is not None:
            self._preview.setPixmap(_bgr_to_pixmap(img, max_w=640))
        elif pipe is not None and pipe.get_state().frame is not None:
            # Fallback: tracker hasn't produced an annotated snapshot yet,
            # but the pipeline has a raw frame. Show that so the user sees
            # the camera is alive.
            self._preview.setPixmap(_bgr_to_pixmap(pipe.get_state().frame, max_w=640))
        if pipe is not None:
            self._fps_card.set_value(f"{pipe.get_state().fps:.1f}")


# ── Motor control tab ──────────────────────────────────────────────────────

class MotorControlTab(QWidget):
    """Right-drag on the 3D head to command pan/tilt; left-drag rotates the
    camera view. Same `HeadPreviewWidget` the calibration wizard uses.

    If Face tracking is running, we borrow its live HeadController via
    SharedHead — dragging here temporarily overrides the tracker output.
    Otherwise we connect our own controller (sim by default).

    Pan/tilt are clamped to cfg.motor.pan_limits_deg / tilt_limits_deg
    before being sent, so the drag can never exceed the soft limits.
    """

    def __init__(
        self,
        cfg: AppConfig,
        shared_head: Optional["SharedHead"] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._shared_head = shared_head
        self._own_head = None

        self._pan_lim = tuple(cfg.motor.pan_limits_deg)
        self._tilt_lim = tuple(cfg.motor.tilt_limits_deg)
        # Target we've commanded (not what the head has reached yet).
        self._target_pan = 0.0
        self._target_tilt = 0.0

        s = _Scaffold(
            self, "Head motor · manual drive",
            "Right-click-drag on the 3D head to move it. Left-drag orbits "
            "the camera view. Soft limits from config.yaml are enforced. "
            "If Face tracking is running, this borrows its motors.",
        )

        self._sim = QCheckBox("Sim motors (safe)")
        self._sim.setChecked(True)
        self._connect_btn = QPushButton("Connect")
        self._disconnect_btn = QPushButton("Disconnect"); self._disconnect_btn.setEnabled(False)
        self._reset_btn = QPushButton("Reset to centre")
        self._reset_btn.setEnabled(False)
        self._connect_btn.clicked.connect(self._connect)
        self._disconnect_btn.clicked.connect(self._disconnect)
        self._reset_btn.clicked.connect(self._reset_centre)

        self._target_card = MetricCard("Target", "—")
        self._actual_card = MetricCard("Actual", "—")

        limits_lbl = QLabel(
            f"Pan: {self._pan_lim[0]:+.1f}° – {self._pan_lim[1]:+.1f}°   "
            f"Tilt: {self._tilt_lim[0]:+.1f}° – {self._tilt_lim[1]:+.1f}°"
        )
        limits_lbl.setObjectName("Subtle"); limits_lbl.setWordWrap(True)

        s.ll.addWidget(self._sim)
        s.ll.addLayout(_btn_row(self._connect_btn, self._disconnect_btn))
        s.ll.addWidget(self._reset_btn)
        s.ll.addWidget(QLabel("Soft limits:"))
        s.ll.addWidget(limits_lbl)
        metrics = QHBoxLayout(); metrics.setSpacing(8)
        metrics.addWidget(self._target_card); metrics.addWidget(self._actual_card)
        s.ll.addLayout(metrics)

        # 3D head preview (falls back to 2D pan + tilt dials on GL failure).
        from companion.ui.widgets.head_preview import HeadPreviewWidget
        self._head_view = HeadPreviewWidget()
        self._head_view.set_limits(self._pan_lim, self._tilt_lim)
        self._head_view.head_drag_delta.connect(self._on_drag_delta)
        s.lr.addWidget(self._head_view, 1)

        self._scaffold = s
        s.finalize()

        # Poll actual pose ~10 Hz so the nose tracks reality.
        self._poll = QTimer(self); self._poll.timeout.connect(self._poll_actual)

    def set_status(self, t: str) -> None:
        self._scaffold.status.setText(t)

    # --- head acquisition ---

    def _active_head(self):
        """Prefer a shared (tracker-owned) head; fall back to our own."""
        if self._shared_head is not None and self._shared_head.head is not None:
            return self._shared_head.head
        return self._own_head

    def _clamp(self, pan: float, tilt: float) -> tuple[float, float]:
        pan = max(self._pan_lim[0], min(self._pan_lim[1], pan))
        tilt = max(self._tilt_lim[0], min(self._tilt_lim[1], tilt))
        return pan, tilt

    def _connect(self) -> None:
        # Borrow the tracker's head if one is live — no need to open our own.
        if self._shared_head is not None and self._shared_head.head is not None:
            self.set_status("Borrowing Face-tracker's head controller.")
            self._after_connect(borrowed=True)
            return

        try:
            from companion.motor.controller import HeadController
            motor_cfg = _copy.deepcopy(self.cfg.motor)
            motor_cfg.sim_only = self._sim.isChecked()
            head = HeadController(motor_cfg)
            head.connect()
            head.enable_torque(True)
            self._own_head = head
        except Exception as exc:
            self.set_status(f"Connect failed: {exc}")
            return
        mode = "SIM" if self._sim.isChecked() else "REAL"
        self.set_status(f"Connected ({mode}).")
        self._after_connect(borrowed=False)

    def _after_connect(self, *, borrowed: bool) -> None:
        self._connect_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(True)
        self._reset_btn.setEnabled(True)
        # Lock the sim switch while connected — can't change mid-session.
        self._sim.setEnabled(False)
        self._poll.start(100)
        head = self._active_head()
        try:
            p, t = head.get_head_pose()
            self._target_pan, self._target_tilt = self._clamp(p, t)
            self._head_view.set_pose(p, t, self._target_pan, self._target_tilt)
            self._target_card.set_value(f"{self._target_pan:+.1f}°/{self._target_tilt:+.1f}°")
            self._actual_card.set_value(f"{p:+.1f}°/{t:+.1f}°")
        except Exception:
            pass

    def _disconnect(self) -> None:
        self._poll.stop()
        if self._own_head is not None:
            try: self._own_head.disconnect()
            except Exception: pass
            self._own_head = None
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._reset_btn.setEnabled(False)
        self._sim.setEnabled(True)
        self._actual_card.set_value("—")
        self.set_status("Disconnected.")

    # --- drag / reset → motor ---

    def _on_drag_delta(self, d_pan: float, d_tilt: float) -> None:
        """HeadPreviewWidget emits (pan_delta, tilt_delta) in degrees on
        each right-button mouse-move. We accumulate onto our current target,
        clamp, and forward to the head controller."""
        new_pan, new_tilt = self._clamp(
            self._target_pan + d_pan,
            self._target_tilt + d_tilt,
        )
        self._command(new_pan, new_tilt)

    def _reset_centre(self) -> None:
        self._command(0.0, 0.0)

    def _command(self, pan_deg: float, tilt_deg: float) -> None:
        self._target_pan, self._target_tilt = self._clamp(pan_deg, tilt_deg)
        self._target_card.set_value(
            f"{self._target_pan:+.1f}°/{self._target_tilt:+.1f}°"
        )
        head = self._active_head()
        if head is None:
            # Update visual target even when not connected (preview-only mode).
            self._head_view.set_pose(
                self._target_pan, self._target_tilt,
                self._target_pan, self._target_tilt,
            )
            self.set_status("Not connected — click Connect to drive motors.")
            return
        try:
            head.set_head_pose(self._target_pan, self._target_tilt)
        except Exception as exc:
            self.set_status(f"Move failed: {exc}")

    def _poll_actual(self) -> None:
        head = self._active_head()
        if head is None:
            return
        try:
            p, t = head.get_head_pose()
        except Exception:
            return
        self._head_view.set_pose(p, t, self._target_pan, self._target_tilt)
        self._actual_card.set_value(f"{p:+.1f}°/{t:+.1f}°")

    def _stop(self) -> None:
        # Called by DebugGUI.closeEvent — also the normal "tab teardown".
        self._disconnect()


# ── Main window ─────────────────────────────────────────────────────────────

_TAB_ORDER = (
    # Input
    ("Audio",         AudioTab),
    ("STT",           STTTab),
    # Reasoning
    ("LLM",           LLMTab),
    ("Tools",         ToolsTab),
    ("Memory",        MemoryTab),
    # Output
    ("TTS",           TTSTab),
    # Vision
    ("Vision",        VisionTab),
    ("VLM",           VLMTab),
    # Embodiment
    ("Face display",  FaceTab),
    ("Face tracking", FaceTrackingTab),
    ("Motor",         MotorControlTab),
)


class DebugGUI(QMainWindow):
    def __init__(
        self,
        cfg: AppConfig,
        conversation=None,
        emotion=None,
        respeaker=None,
        scene=None,
    ) -> None:
        super().__init__()
        # Deep-copy the config so that individual tabs mutating cfg fields
        # (STT backend, TTS engine/voice, LLM model, DOA offset) don't
        # corrupt a parent app's live state. The one exception is the
        # on-disk config.yaml — AudioTab's _persist_doa_offset writes
        # directly to the file so the offset survives restarts.
        self.cfg = _copy.deepcopy(cfg)
        self.setWindowTitle("Companion · Debug")
        self.resize(1240, 820)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self.setCentralWidget(self._tabs)

        # Section labels inside a "Card" QFrame were picking up the global
        # QWidget bg colour, rendering as dark horizontal strips. Force them
        # transparent so they blend into the card surface.
        self.setStyleSheet("QLabel { background: transparent; }")

        # Vision and Face tracking share one EmotionPipeline (one camera).
        # Either tab can run alone, or both simultaneously.
        self.shared_vision = SharedVision(cfg, preexisting=emotion)
        # Audio-tab DOA calibration reads the current head pose while Face
        # tracking is running — this is the handoff point.
        self.shared_head = SharedHead()
        # Vision tab mirrors the detected emotion onto the Face display
        # renderer in real time — this is the handoff point.
        self.shared_renderer = SharedRenderer()

        for label, klass in _TAB_ORDER:
            if klass is VisionTab:
                widget = VisionTab(
                    cfg,
                    shared_vision=self.shared_vision,
                    shared_renderer=self.shared_renderer,
                )
            elif klass is FaceTrackingTab:
                widget = FaceTrackingTab(
                    cfg,
                    shared_vision=self.shared_vision,
                    shared_head=self.shared_head,
                )
            elif klass is AudioTab:
                widget = AudioTab(cfg, shared_head=self.shared_head)
            elif klass is MotorControlTab:
                widget = MotorControlTab(cfg, shared_head=self.shared_head)
            elif klass is FaceTab:
                widget = FaceTab(cfg, shared_renderer=self.shared_renderer)
            else:
                widget = klass(cfg)
            self._tabs.addTab(widget, label)

    def closeEvent(self, event) -> None:
        """Stop every tab's work before the window closes.
        Otherwise pipelines/servos/audio streams leak until Python exits."""
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            stop = getattr(w, "_stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc:
                    log.debug(f"tab {self._tabs.tabText(i)} stop failed: {exc!r}")
        super().closeEvent(event)


def launch(
    cfg=None,
    conversation=None,
    emotion=None,
    respeaker=None,
    scene=None,
) -> None:
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
