"""
Live valence-arousal circumplex visualizer.

Run as a standalone window:

    python3 -m companion.vision.visualizer

Renders:
  - Russell's circumplex (unit circle) with the 8 emotion anchors on the rim
  - Quadrant labels: Happy/Excited, Angry/Tense, Sad/Bored, Calm/Relaxed
  - A glowing dot at the current (valence, arousal), with a fading trail
  - Probability bars for the 8 emotions
  - Camera preview thumbnail with the face bbox overlaid
  - FPS / latency / GPU provider readout
"""

import math
import os
import sys
from collections import deque
from typing import Deque, Tuple

import numpy as np
import yaml

from .emotion_classifier import EMOTION_LABELS, EMOTION_VA
from .emotion_pipeline import EmotionPipeline


def _load_vision_config() -> dict:
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config.yaml",
    )
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("vision", {}) or {}


def main() -> int:
    # Imported here so the rest of the package can be used headless without PyQt.
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtGui, QtWidgets

    pg.setConfigOptions(antialias=True, background="#101018", foreground="#e8e8f0")

    vision_cfg = _load_vision_config()
    pipeline = EmotionPipeline(vision_cfg)
    pipeline.start()

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Emotion Circumplex — Live")
    win.resize(1200, 720)

    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    layout = QtWidgets.QGridLayout(central)

    # ─── Circumplex plot ───
    plot = pg.PlotWidget()
    plot.setAspectLocked(True)
    plot.setXRange(-1.15, 1.15)
    plot.setYRange(-1.15, 1.15)
    plot.showGrid(x=False, y=False)
    plot.setLabel("bottom", "Valence  (− unpleasant      pleasant +)")
    plot.setLabel("left", "Arousal  (− calm                 excited +)")
    layout.addWidget(plot, 0, 0, 2, 1)

    # Unit circle
    theta = np.linspace(0, 2 * math.pi, 256)
    plot.plot(np.cos(theta), np.sin(theta), pen=pg.mkPen("#444466", width=2))
    # Axes through origin
    plot.plot([-1, 1], [0, 0], pen=pg.mkPen("#33334d", width=1))
    plot.plot([0, 0], [-1, 1], pen=pg.mkPen("#33334d", width=1))

    # Emotion anchors
    for label, (v, a) in EMOTION_VA.items():
        anchor = pg.ScatterPlotItem(
            [v], [a], size=12, brush=pg.mkBrush("#6688ff"), pen=pg.mkPen("#aabbff", width=1)
        )
        plot.addItem(anchor)
        text = pg.TextItem(label, color="#aabbff", anchor=(0.5, 1.4))
        text.setPos(v, a)
        plot.addItem(text)

    # Quadrant labels
    quad = [
        (" Happy / Excited",  0.95,  0.95, "#88ffaa"),
        (" Angry / Tense",   -0.95,  0.95, "#ff8888"),
        (" Sad / Bored",     -0.95, -0.95, "#88aaff"),
        (" Calm / Relaxed",   0.95, -0.95, "#aaffee"),
    ]
    for text, x, y, color in quad:
        t = pg.TextItem(text, color=color, anchor=(0.5 if x > 0 else 0.5, 0.5))
        t.setPos(x, y)
        plot.addItem(t)

    # Trail + current point
    trail: Deque[Tuple[float, float]] = deque(maxlen=40)
    trail_curve = plot.plot([], [], pen=pg.mkPen((255, 200, 80, 160), width=2))
    current = pg.ScatterPlotItem(
        [], [], size=24, brush=pg.mkBrush(255, 220, 80, 230),
        pen=pg.mkPen("#ffffff", width=2),
    )
    plot.addItem(current)

    # ─── Probability bars ───
    bars_widget = pg.PlotWidget()
    bars_widget.setBackground("#101018")
    bars_widget.setYRange(0, 1)
    bars_widget.setMouseEnabled(x=False, y=False)
    bars_widget.getAxis("bottom").setTicks(
        [list(zip(range(len(EMOTION_LABELS)), EMOTION_LABELS))]
    )
    bars_widget.setLabel("left", "Probability")
    bar_item = pg.BarGraphItem(
        x=list(range(len(EMOTION_LABELS))),
        height=[0] * len(EMOTION_LABELS),
        width=0.7,
        brush="#66aaff",
    )
    bars_widget.addItem(bar_item)
    layout.addWidget(bars_widget, 1, 1)

    # ─── Camera preview ───
    preview = QtWidgets.QLabel()
    preview.setFixedSize(480, 270)
    preview.setStyleSheet("background:#000; border:1px solid #333;")
    preview.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(preview, 0, 1)

    # ─── Status readout ───
    status = QtWidgets.QLabel("Starting…")
    status.setStyleSheet("color:#dddde8; font: 14px monospace; padding:6px;")
    layout.addWidget(status, 2, 0, 1, 2)

    layout.setColumnStretch(0, 3)
    layout.setColumnStretch(1, 2)

    # ─── Update tick ───
    def tick() -> None:
        st = pipeline.get_state()
        if st.has_face:
            trail.append((st.valence, st.arousal))
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            trail_curve.setData(xs, ys)
            current.setData([st.valence], [st.arousal])
        bar_item.setOpts(height=list(st.probs))

        # Camera preview
        if st.frame is not None:
            import cv2
            f = st.frame
            if st.bbox is not None:
                x, y, w, h = st.bbox
                cv2.rectangle(f, (x, y), (x + w, y + h), (80, 220, 255), 2)
                cv2.putText(
                    f, f"{st.label} {st.confidence*100:.0f}%",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (80, 220, 255), 2,
                )
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w, _ = f.shape
            qimg = QtGui.QImage(f.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                preview.width(), preview.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
            )
            preview.setPixmap(pix)

        providers = ",".join(pipeline.classifier.providers)
        face_txt = "FACE" if st.has_face else "no face"
        status.setText(
            f" {face_txt}  |  emotion: {st.label:<10} ({st.confidence*100:5.1f}%)  "
            f"|  valence={st.valence:+.2f}  arousal={st.arousal:+.2f}  "
            f"|  {st.fps:5.1f} fps  {st.latency_ms:5.1f} ms  |  ORT: {providers}"
        )

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(33)  # ~30 Hz UI refresh

    win.show()
    try:
        ret = app.exec_()
    finally:
        pipeline.stop()
    return ret


if __name__ == "__main__":
    sys.exit(main())
