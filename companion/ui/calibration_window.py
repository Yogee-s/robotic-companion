"""Calibration wizard — configures the two ST3215 servos in the differential
bevel head. Works equally well with a live ST3215Bus or a SimulatedBus.

Wizard flow (see /home/yogee/.claude/plans/okay-i-have-a-crispy-papert.md):
  1. Connect + scan + (optional) ID assignment
  2. Per-motor direction test
  3. Rough zero (torque off, hand-hold center)
  4. Combined pan/tilt sanity
  5. Limit discovery (4 jog-to-stop directions)
  6. Refine zeros (pan midpoint; tilt user-set)
  7. Gear ratio measurement (empirical)
  8. Verify (live preview + sliders)
  9. Save to config.yaml
"""

from __future__ import annotations

import copy
import glob
import logging
import math
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget, QWizard, QWizardPage,
)

from companion.core.config import MotorConfig
from companion.motor.bus import ST3215Bus, ServoBus, SimulatedBus
from companion.motor.calibration import (
    CalibrationResult, calibration_summary, save_to_config_yaml,
)
from companion.motor.controller import HeadController
from companion.motor.kinematics import degrees_to_ticks, ticks_to_degrees
from companion.ui.widgets.head_preview import HeadPreviewWidget

log = logging.getLogger(__name__)


# ── Shared wizard state ─────────────────────────────────────────────────────

@dataclass
class WizardState:
    cfg: MotorConfig                                           # working copy, mutated as user proceeds
    yaml_path: str
    controller: Optional[HeadController] = None
    # raw limit captures (motor-tick space, before zero refinement)
    pan_min_raw_l: Optional[int] = None
    pan_min_raw_r: Optional[int] = None
    pan_max_raw_l: Optional[int] = None
    pan_max_raw_r: Optional[int] = None
    tilt_min_raw_l: Optional[int] = None
    tilt_min_raw_r: Optional[int] = None
    tilt_max_raw_l: Optional[int] = None
    tilt_max_raw_r: Optional[int] = None


# ── Wizard shell ────────────────────────────────────────────────────────────

class CalibrationWizard(QWizard):
    PAGE_CONNECT = 0
    PAGE_DIRECTION = 1
    PAGE_ROUGH_ZERO = 2
    PAGE_COMBINED = 3
    PAGE_LIMITS = 4
    PAGE_REFINE = 5
    PAGE_GEAR = 6
    PAGE_VERIFY = 7
    PAGE_SAVE = 8

    def __init__(self, cfg: MotorConfig, yaml_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Head Motor Calibration")
        self.setOption(QWizard.IndependentPages, False)
        self.setWizardStyle(QWizard.ModernStyle)
        self.resize(1000, 700)

        self.state = WizardState(cfg=copy.deepcopy(cfg), yaml_path=yaml_path)
        # Calibration involves intentionally driving toward mechanical limits
        # and doing test sweeps that can exceed the stall-detect 1.5 s window.
        # Disable stall detection wizard-wide so the controller doesn't
        # silently latch a "hold at current" mid-move. The original config
        # value will be loaded back from disk when calibration ends.
        self.state.cfg.stall_detect = False

        self.setPage(self.PAGE_CONNECT, ConnectPage(self.state))
        self.setPage(self.PAGE_DIRECTION, DirectionTestPage(self.state))
        self.setPage(self.PAGE_ROUGH_ZERO, RoughZeroPage(self.state))
        self.setPage(self.PAGE_COMBINED, CombinedSanityPage(self.state))
        self.setPage(self.PAGE_LIMITS, LimitDiscoveryPage(self.state))
        self.setPage(self.PAGE_REFINE, RefineZeroPage(self.state))
        self.setPage(self.PAGE_GEAR, GearRatioPage(self.state))
        self.setPage(self.PAGE_VERIFY, VerifyPage(self.state))
        self.setPage(self.PAGE_SAVE, SavePage(self.state))

    def closeEvent(self, event):
        if self.state.controller is not None:
            try:
                self.state.controller.disconnect()
            except Exception as e:
                log.warning(f"wizard close cleanup: {e}")
        super().closeEvent(event)


# ── Page 1: Connect + Scan + Assign ID ──────────────────────────────────────

class ConnectPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("1. Connect & scan the bus")
        self.setSubTitle(
            "Pick the serial port, scan for ST3215 servos, and assign unique "
            "IDs if needed. Simulator mode works without any hardware."
        )
        lay = QVBoxLayout(self)

        form = QFormLayout()
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self._refresh_ports()
        self.port_combo.setCurrentText(state.cfg.port)
        form.addRow("Serial port:", self.port_combo)

        self.sim_check = QCheckBox("Simulator only (no hardware required)")
        self.sim_check.setChecked(state.cfg.sim_only)
        form.addRow(self.sim_check)

        self.left_id_spin = QSpinBox()
        self.left_id_spin.setRange(1, 253)
        self.left_id_spin.setValue(state.cfg.left_servo_id)
        form.addRow("Left servo ID:", self.left_id_spin)

        self.right_id_spin = QSpinBox()
        self.right_id_spin.setRange(1, 253)
        self.right_id_spin.setValue(state.cfg.right_servo_id)
        form.addRow("Right servo ID:", self.right_id_spin)
        lay.addLayout(form)

        btn_row = QHBoxLayout()
        self.scan_btn = QPushButton("Connect && scan")
        self.scan_btn.clicked.connect(self._connect_and_scan)
        btn_row.addWidget(self.scan_btn)

        self.assign_btn = QPushButton("Assign left ID → 1, then right → 2")
        self.assign_btn.setEnabled(False)
        self.assign_btn.clicked.connect(self._assign_ids)
        btn_row.addWidget(self.assign_btn)
        lay.addLayout(btn_row)

        self.status = QLabel("Not connected.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)

        self._connected = False

    def _refresh_ports(self) -> None:
        self.port_combo.clear()
        candidates = sorted(set(
            glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
            + glob.glob("/dev/ttyCH341USB*") + glob.glob("/dev/ttyTHS*")
        ))
        if not candidates:
            candidates = ["/dev/ttyUSB0"]
        for p in candidates:
            self.port_combo.addItem(p)

    def _connect_and_scan(self) -> None:
        cfg = self.state.cfg
        cfg.sim_only = self.sim_check.isChecked()
        cfg.port = self.port_combo.currentText().strip()
        cfg.left_servo_id = self.left_id_spin.value()
        cfg.right_servo_id = self.right_id_spin.value()

        if self.state.controller is not None:
            try:
                self.state.controller.disconnect()
            except Exception:
                pass
            self.state.controller = None

        try:
            ctrl = HeadController(cfg)
            ctrl.connect()
            self.state.controller = ctrl
        except Exception as e:
            self.status.setText(f"<span style='color:#e66'>Connection failed: {e}</span>")
            return

        found = ctrl.bus.scan(range(1, 20))
        if not found:
            self.status.setText("<span style='color:#e66'>No servos found on the bus.</span>")
            return
        needed = {cfg.left_servo_id, cfg.right_servo_id}
        if needed.issubset(set(found)):
            self.status.setText(
                f"<span style='color:#4a4'>Connected.</span> Found IDs: {found}. "
                f"Expected L={cfg.left_servo_id}, R={cfg.right_servo_id} — OK."
            )
            self._connected = True
            self.completeChanged.emit()
        else:
            msg = (
                f"Found IDs {found}, but expected L={cfg.left_servo_id}, "
                f"R={cfg.right_servo_id}.\n"
            )
            if found == [1]:
                msg += (
                    "Looks like fresh-from-factory: both servos default to ID 1, "
                    "or only one is connected. Disconnect one servo, click "
                    "'Assign left ID → 1, then right → 2' to assign the other, "
                    "then reconnect both and re-scan."
                )
                self.assign_btn.setEnabled(True)
            else:
                msg += "Use the spin boxes to match the found IDs, or reassign."
            self.status.setText(msg)

    def _assign_ids(self) -> None:
        """Helper for fresh servos: renames whichever single servo is on the
        bus (both typically default to ID=1) to the right-motor ID."""
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            found = ctrl.bus.scan(range(1, 20))
            if len(found) != 1:
                QMessageBox.warning(
                    self, "Assign ID",
                    f"Expected exactly one servo on the bus for reassignment, "
                    f"found {len(found)} ({found}). Disconnect all but one."
                )
                return
            current = found[0]
            target = self.state.cfg.right_servo_id
            if current == target:
                QMessageBox.information(
                    self, "Assign ID",
                    f"Servo is already at ID {target}. Reconnect the other servo "
                    f"(should be at ID {self.state.cfg.left_servo_id}) and re-scan."
                )
                return
            ctrl.bus.set_id(current, target)
            QMessageBox.information(
                self, "Assign ID",
                f"Servo ID {current} → {target}. Now connect the other servo "
                f"(it should be at ID {self.state.cfg.left_servo_id}) and click Connect && scan."
            )
        except Exception as e:
            QMessageBox.warning(self, "Assign ID", f"Failed: {e}")

    def isComplete(self) -> bool:
        return self._connected


# ── Page 2: Per-motor direction test ────────────────────────────────────────

class DirectionTestPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("2. Direction test")
        self.setSubTitle(
            "Jog each motor a small amount. Watch the pinion (or the mesh "
            "point). Set left_direction / right_direction so that a POSITIVE "
            "jog rotates the small pinion in its reference positive direction."
        )
        lay = QVBoxLayout(self)
        note = QLabel(
            "Tip: if you're not sure what 'positive' should mean, simply ensure "
            "both motors have a consistent convention — the next (combined-sanity) "
            "step checks whether pan/tilt come out right."
        )
        note.setWordWrap(True)
        lay.addWidget(note)

        for side in ("left", "right"):
            box = QFrame()
            box.setFrameShape(QFrame.Box)
            row = QHBoxLayout(box)
            row.addWidget(QLabel(f"<b>{side.capitalize()} motor</b>"))
            jog_plus = QPushButton("Jog +5°")
            jog_minus = QPushButton("Jog −5°")
            flip = QCheckBox("invert (flip direction)")
            flip.setChecked(
                getattr(state.cfg, f"{side}_direction") < 0
            )
            flip.stateChanged.connect(
                lambda checked, s=side: setattr(
                    state.cfg, f"{s}_direction", -1 if checked else 1
                )
            )
            jog_plus.clicked.connect(lambda _, s=side: self._jog(s, +5))
            jog_minus.clicked.connect(lambda _, s=side: self._jog(s, -5))
            row.addWidget(jog_plus)
            row.addWidget(jog_minus)
            row.addWidget(flip)
            row.addStretch(1)
            lay.addWidget(box)

        self.status = QLabel("Both motors enabled. Click Next when direction signs feel right.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)

    def initializePage(self) -> None:
        """Pin each motor's goal to its CURRENT position (folded into 0..4095)
        BEFORE enabling torque. This way, torque-on doesn't trigger a big
        motion toward a stale goal. The software-tracked target is then
        incremented ±57 ticks per jog click — small, predictable moves."""
        ctrl = self.state.controller
        if ctrl is None:
            return
        cfg = self.state.cfg
        self._targets: dict[str, int] = {}
        try:
            for side in ("left", "right"):
                sid = cfg.left_servo_id if side == "left" else cfg.right_servo_id
                pos = int(ctrl.bus.read_position(sid)) % 4096
                ctrl.bus.write_goal(sid, pos)
                self._targets[side] = pos
            ctrl.enable_torque(True)
            self.status.setText(
                f"Torque ON, held at current position. "
                f"L tick={self._targets['left']}, R tick={self._targets['right']}. "
                f"Jogs move ±57 ticks (~5° motor shaft)."
            )
        except Exception as e:
            log.warning(f"direction-test init: {e}")
            self.status.setText(f"init failed: {e}")

    def _jog(self, side: str, deg: float) -> None:
        """Rotate one motor by ±deg around our software-tracked target.
        Uses a persistent per-side target instead of re-reading the encoder,
        because position reads can be noisy or mid-motion and would make
        repeated clicks drift or reverse."""
        ctrl = self.state.controller
        if ctrl is None:
            return
        cfg = self.state.cfg
        sid = cfg.left_servo_id if side == "left" else cfg.right_servo_id
        direction = cfg.left_direction if side == "left" else cfg.right_direction
        if not hasattr(self, "_targets") or side not in self._targets:
            try:
                self._targets = getattr(self, "_targets", {})
                self._targets[side] = ctrl.bus.read_position(sid)
            except Exception as e:
                self.status.setText(f"read_position error: {e}")
                return
        prev = self._targets[side]
        delta_ticks = degrees_to_ticks(deg) * direction
        target = max(0, min(4095, prev + delta_ticks))
        try:
            ctrl.bus.write_goal(sid, target)
            self._targets[side] = target
            self.status.setText(
                f"Jogged {side} by {deg:+.1f}° (motor-shaft). Tick: {prev} → {target}."
            )
        except Exception as e:
            self.status.setText(f"write_goal error: {e}")


# ── Page 3: Rough zero ──────────────────────────────────────────────────────

class RoughZeroPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("3. Rough zero")
        self.setSubTitle(
            "Torque is now OFF so you can hand-hold the head. Position the head "
            "so it's approximately looking forward and level. Click 'Capture' "
            "when holding it there. This is a temporary reference — later steps "
            "will compute the true center from the discovered mechanical limits."
        )
        lay = QVBoxLayout(self)

        self.capture_btn = QPushButton("Capture rough zero (from current encoder position)")
        self.capture_btn.clicked.connect(self._capture)
        lay.addWidget(self.capture_btn)

        self.readout = QLabel("Left tick: —    Right tick: —")
        lay.addWidget(self.readout)

        self.status = QLabel("Move head to forward+level, then click Capture.")
        lay.addWidget(self.status)
        lay.addStretch(1)

        self._captured = False

    def initializePage(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            ctrl.enable_torque(False)
        except Exception as e:
            log.warning(f"rough-zero disable torque: {e}")

    def _capture(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            lt, rt = ctrl.read_raw_ticks()
        except Exception as e:
            self.status.setText(f"Failed to read: {e}")
            return
        self.state.cfg.left_zero_tick = lt
        self.state.cfg.right_zero_tick = rt
        self.readout.setText(f"Left tick: {lt}    Right tick: {rt}")
        self.status.setText(
            "Rough zero captured. You can now continue — next steps use this "
            "as the frame of reference for jogging."
        )
        self._captured = True
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._captured


# ── Page 4: Combined sanity ─────────────────────────────────────────────────

class CombinedSanityPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("4. Combined pan/tilt sanity")
        self.setSubTitle(
            "Compose pan and tilt commands and watch the head. If 'pan left' "
            "actually tilts, or goes the wrong way, toggle the axis inversions."
        )
        main = QHBoxLayout(self)

        left = QVBoxLayout()
        for label, pan, tilt in [
            ("Pan LEFT +10°", -10, 0), ("Pan RIGHT +10°", +10, 0),
            ("Tilt UP +5°", 0, +5),    ("Tilt DOWN +5°", 0, -5),
            ("Center (0, 0)", 0, 0),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, p=pan, t=tilt: self._move(p, t))
            left.addWidget(btn)

        left.addSpacing(10)
        self.invert_pan = QCheckBox("Invert pan axis")
        self.invert_pan.setChecked(state.cfg.invert_pan)
        self.invert_pan.stateChanged.connect(
            lambda s: setattr(state.cfg, "invert_pan", bool(s))
        )
        self.invert_tilt = QCheckBox("Invert tilt axis")
        self.invert_tilt.setChecked(state.cfg.invert_tilt)
        self.invert_tilt.stateChanged.connect(
            lambda s: setattr(state.cfg, "invert_tilt", bool(s))
        )
        left.addWidget(self.invert_pan)
        left.addWidget(self.invert_tilt)
        left.addStretch(1)
        main.addLayout(left, stretch=0)

        self.preview = HeadPreviewWidget()
        self.preview.set_limits(
            tuple(state.cfg.pan_limits_deg), tuple(state.cfg.tilt_limits_deg)
        )
        main.addWidget(self.preview, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_preview)
        self._timer.setInterval(int(1000 / max(1, state.cfg.poll_hz)))

    def initializePage(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            ctrl.enable_torque(True)
        except Exception as e:
            log.warning(f"combined-sanity enable torque: {e}")
        self._timer.start()

    def cleanupPage(self) -> None:
        self._timer.stop()

    def _move(self, pan: float, tilt: float) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            ctrl.set_head_pose(pan, tilt)
        except Exception as e:
            log.warning(f"combined move: {e}")

    def _refresh_preview(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        pan, tilt = ctrl.state.pan_deg, ctrl.state.tilt_deg
        self.preview.set_pose(pan, tilt, ctrl.state.target_pan_deg, ctrl.state.target_tilt_deg)


# ── Page 5: Limit discovery ─────────────────────────────────────────────────

class LimitDiscoveryPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("5. Limit discovery")
        self.setSubTitle(
            "Step-jog each axis until the head just touches the mechanical "
            "limit, then click 'Mark limit'. Do all four: pan right, pan left, "
            "tilt up, tilt down. A 2° safety margin will be applied inward."
        )
        main = QHBoxLayout(self)

        controls = QVBoxLayout()

        # Show whatever limits were already saved in config.yaml from a
        # previous calibration — useful so the user knows what they're
        # overriding (or can choose to keep by re-marking the same way).
        self.saved_label = QLabel("(saved limits will appear here)")
        self.saved_label.setWordWrap(True)
        font = self.saved_label.font()
        font.setItalic(True)
        self.saved_label.setFont(font)
        controls.addWidget(self.saved_label)

        controls.addSpacing(4)
        self.current_label = QLabel("At pose (0.0°, 0.0°)")
        controls.addWidget(self.current_label)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step size (deg):"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 10.0)
        self.step_spin.setSingleStep(0.5)
        self.step_spin.setValue(2.0)
        step_row.addWidget(self.step_spin)
        controls.addLayout(step_row)

        jog_grid = QGridLayout()
        self.pan_jog_left = QPushButton("Jog PAN −")
        self.pan_jog_right = QPushButton("Jog PAN +")
        self.tilt_jog_up = QPushButton("Jog TILT +")
        self.tilt_jog_down = QPushButton("Jog TILT −")
        self.center_btn = QPushButton("⌂ Center (0°, 0°)")
        for btn in (self.pan_jog_left, self.pan_jog_right,
                    self.tilt_jog_down, self.tilt_jog_up, self.center_btn):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.pan_jog_left.clicked.connect(lambda: self._jog(-1, 0))
        self.pan_jog_right.clicked.connect(lambda: self._jog(+1, 0))
        self.tilt_jog_up.clicked.connect(lambda: self._jog(0, +1))
        self.tilt_jog_down.clicked.connect(lambda: self._jog(0, -1))
        self.center_btn.clicked.connect(self._center)
        jog_grid.addWidget(self.pan_jog_left, 0, 0)
        jog_grid.addWidget(self.pan_jog_right, 0, 1)
        jog_grid.addWidget(self.tilt_jog_down, 1, 0)
        jog_grid.addWidget(self.tilt_jog_up, 1, 1)
        jog_grid.addWidget(self.center_btn, 2, 0, 1, 2)   # span both columns
        controls.addLayout(jog_grid)

        controls.addSpacing(6)

        self.status_labels: dict[str, QLabel] = {}
        for key, label in [
            ("pan_min", "Pan MIN (left)"),
            ("pan_max", "Pan MAX (right)"),
            ("tilt_min", "Tilt MIN (down)"),
            ("tilt_max", "Tilt MAX (up)"),
        ]:
            block = QVBoxLayout()
            row = QHBoxLayout()
            mark_btn = QPushButton(f"Mark {label}")
            mark_btn.clicked.connect(lambda _, k=key: self._mark(k))
            mark_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row.addWidget(mark_btn, stretch=2)
            test_btn = QPushButton("Test ▶")
            test_btn.setEnabled(False)
            test_btn.clicked.connect(lambda _, k=key: self._test_limit(k))
            test_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row.addWidget(test_btn, stretch=1)
            block.addLayout(row)
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            font = lbl.font()
            font.setFamily("Monospace")
            lbl.setFont(font)
            block.addWidget(lbl)
            self.status_labels[key] = lbl
            setattr(self, f"_test_btn_{key}", test_btn)
            controls.addLayout(block)

        controls.addSpacing(8)
        self.live_status = QLabel("(live status — appears once moving)")
        self.live_status.setWordWrap(True)
        font = self.live_status.font()
        font.setFamily("Monospace")
        self.live_status.setFont(font)
        controls.addWidget(self.live_status)

        controls.addStretch(1)
        main.addLayout(controls, stretch=0)

        self.preview = HeadPreviewWidget()
        self.preview.set_limits(
            tuple(state.cfg.pan_limits_deg), tuple(state.cfg.tilt_limits_deg)
        )
        main.addWidget(self.preview, stretch=1)

        self._pan = 0.0
        self._tilt = 0.0
        self._marks: dict[str, tuple[int, int]] = {}

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_preview)
        self._timer.setInterval(int(1000 / max(1, state.cfg.poll_hz)))

    def initializePage(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        # Display the previously-saved limits so the user can decide whether
        # to keep them (just press Next or Test) or change them (re-jog and
        # re-mark). They were loaded from config.yaml at wizard startup.
        self._saved_pan_limits = list(self.state.cfg.pan_limits_deg)
        self._saved_tilt_limits = list(self.state.cfg.tilt_limits_deg)
        self.saved_label.setText(
            f"Saved limits in config.yaml:\n"
            f"  pan  = {self._saved_pan_limits[0]:+.1f}°  ..  "
            f"{self._saved_pan_limits[1]:+.1f}°\n"
            f"  tilt = {self._saved_tilt_limits[0]:+.1f}°  ..  "
            f"{self._saved_tilt_limits[1]:+.1f}°"
        )
        # Pre-fill the marks from the saved limit poses, BEFORE we widen the
        # runtime clamps — kinematics uses left/right_zero_tick + gear_ratio
        # to compute the corresponding raw tick values for each marked corner.
        # This means: open step 5, see saved values populated, can immediately
        # Test each limit or just hit Next. Re-jogging is optional.
        from companion.motor.kinematics import head_pose_to_ticks
        try:
            cfg = self.state.cfg
            corners = {
                "pan_min": (self._saved_pan_limits[0], 0.0),
                "pan_max": (self._saved_pan_limits[1], 0.0),
                "tilt_min": (0.0, self._saved_tilt_limits[0]),
                "tilt_max": (0.0, self._saved_tilt_limits[1]),
            }
            self._marks.clear()
            s = self.state
            for key, (pan, tilt) in corners.items():
                lt, rt = head_pose_to_ticks(pan, tilt, cfg)
                self._marks[key] = (lt, rt, pan, tilt)
                if key == "pan_min":
                    s.pan_min_raw_l, s.pan_min_raw_r = lt, rt
                elif key == "pan_max":
                    s.pan_max_raw_l, s.pan_max_raw_r = lt, rt
                elif key == "tilt_min":
                    s.tilt_min_raw_l, s.tilt_min_raw_r = lt, rt
                elif key == "tilt_max":
                    s.tilt_max_raw_l, s.tilt_max_raw_r = lt, rt
                self.status_labels[key].setText(
                    f"L={lt} R={rt}   (pose {pan:+.1f}°, {tilt:+.1f}°)   [from saved]"
                )
                getattr(self, f"_test_btn_{key}").setEnabled(True)
        except Exception as e:
            log.warning(f"prefill saved limits: {e}")
            for key in ("pan_min", "pan_max", "tilt_min", "tilt_max"):
                getattr(self, f"_test_btn_{key}").setEnabled(False)
                self.status_labels[key].setText("—")

        # Now widen the runtime soft limits so the user can re-jog past the
        # saved values if they want to discover wider mechanical limits.
        # Step 6 recomputes from whatever marks are in state when we leave.
        self.state.cfg.pan_limits_deg = [-180.0, 180.0]
        self.state.cfg.tilt_limits_deg = [-90.0, 90.0]
        try:
            ctrl.enable_torque(True)
            ctrl.bus.set_goal_speed(self.state.cfg.left_servo_id, 500)
            ctrl.bus.set_goal_speed(self.state.cfg.right_servo_id, 500)
        except Exception as e:
            log.warning(f"limit-discovery init: {e}")
        self._pan = 0.0
        self._tilt = 0.0
        self._timer.start()
        # Marks are pre-populated, so Next is enabled immediately
        self.completeChanged.emit()

    def cleanupPage(self) -> None:
        self._timer.stop()

    def _jog(self, pan_dir: int, tilt_dir: int) -> None:
        step = self.step_spin.value()
        self._pan += pan_dir * step
        self._tilt += tilt_dir * step
        # Hard-cap at 2× soft limits so an errant step can't go wild
        self._pan = max(-180, min(180, self._pan))
        self._tilt = max(-90, min(90, self._tilt))
        self.current_label.setText(f"At pose ({self._pan:+.1f}°, {self._tilt:+.1f}°)")
        ctrl = self.state.controller
        if ctrl is not None:
            try:
                ctrl.set_head_pose(self._pan, self._tilt)
            except Exception as e:
                log.warning(f"limit jog: {e}")

    def _mark(self, key: str) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            lt, rt = ctrl.read_raw_ticks()
        except Exception as e:
            log.warning(f"mark read: {e}")
            return
        self._marks[key] = (lt, rt, self._pan, self._tilt)
        s = self.state
        if key == "pan_min":
            s.pan_min_raw_l, s.pan_min_raw_r = lt, rt
        elif key == "pan_max":
            s.pan_max_raw_l, s.pan_max_raw_r = lt, rt
        elif key == "tilt_min":
            s.tilt_min_raw_l, s.tilt_min_raw_r = lt, rt
        elif key == "tilt_max":
            s.tilt_max_raw_l, s.tilt_max_raw_r = lt, rt
        self.status_labels[key].setText(
            f"L={lt} R={rt}   (pose {self._pan:+.1f}°, {self._tilt:+.1f}°)"
        )
        # Enable the corresponding Test button now that a limit is captured
        getattr(self, f"_test_btn_{key}").setEnabled(True)
        self.completeChanged.emit()

    def _center(self) -> None:
        """Drive both axes back to (0°, 0°) — the rough-zero captured in step 3."""
        self._pan = 0.0
        self._tilt = 0.0
        self.current_label.setText(f"At pose ({self._pan:+.1f}°, {self._tilt:+.1f}°)")
        ctrl = self.state.controller
        if ctrl is not None:
            try:
                ctrl.set_head_pose(0.0, 0.0)
            except Exception as e:
                log.warning(f"center: {e}")

    def _test_limit(self, key: str) -> None:
        """Drive the head back to a previously-marked limit so you can verify
        it visually. Bumps speed up to 2000 ticks/s for the move (the 500 used
        for jogging is too slow — large moves time out the 1.5 s stall window
        mid-flight, which is why a single Test press would stop halfway).
        Restores the slower jog speed when done."""
        if key not in self._marks:
            return
        _lt, _rt, pan, tilt = self._marks[key]
        self._pan = pan
        self._tilt = tilt
        self.current_label.setText(f"At pose ({self._pan:+.1f}°, {self._tilt:+.1f}°)")
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            ctrl.bus.set_goal_speed(self.state.cfg.left_servo_id, 2000)
            ctrl.bus.set_goal_speed(self.state.cfg.right_servo_id, 2000)
            ctrl.set_head_pose(pan, tilt)
        except Exception as e:
            log.warning(f"test limit {key}: {e}")
            return
        # Drop speed back to jog default once the move has had time to start
        QTimer.singleShot(2500, self._restore_jog_speed)

    def _restore_jog_speed(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            ctrl.bus.set_goal_speed(self.state.cfg.left_servo_id, 500)
            ctrl.bus.set_goal_speed(self.state.cfg.right_servo_id, 500)
        except Exception as e:
            log.warning(f"restore jog speed: {e}")

    def _refresh_preview(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        self.preview.set_pose(
            ctrl.state.pan_deg, ctrl.state.tilt_deg,
            ctrl.state.target_pan_deg, ctrl.state.target_tilt_deg,
        )
        # Live diagnostic: shows commanded vs actual ticks plus stall state.
        # Helps distinguish "head hit a real mechanical stop" from
        # "encoder ran out of range" from "stall detector fired prematurely".
        cfg = self.state.cfg
        l_goal = ctrl.state.left_goal_tick
        r_goal = ctrl.state.right_goal_tick
        l_act = ctrl.state.left.position_tick if ctrl.state.left else 0
        r_act = ctrl.state.right.position_tick if ctrl.state.right else 0
        l_err = l_act - l_goal
        r_err = r_act - r_goal
        l_at_min = l_act <= 5
        l_at_max = l_act >= 4090
        r_at_min = r_act <= 5
        r_at_max = r_act >= 4090
        flags = []
        if ctrl.state.stalled:
            flags.append("STALLED")
        if l_at_min:  flags.append("L@tick0")
        if l_at_max:  flags.append("L@tick4095")
        if r_at_min:  flags.append("R@tick0")
        if r_at_max:  flags.append("R@tick4095")
        flag_str = "  ⚠ " + " ".join(flags) if flags else ""
        self.live_status.setText(
            f"actual: pan={ctrl.state.pan_deg:+6.1f}° tilt={ctrl.state.tilt_deg:+6.1f}°\n"
            f"L: tick={l_act:4d} (goal {l_goal:4d}, err {l_err:+5d})\n"
            f"R: tick={r_act:4d} (goal {r_goal:4d}, err {r_err:+5d})"
            f"{flag_str}"
        )

    def isComplete(self) -> bool:
        return all(k in self._marks for k in ("pan_min", "pan_max", "tilt_min", "tilt_max"))


# ── Page 6: Refine zeros ────────────────────────────────────────────────────

class RefineZeroPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("6. Refine zeros")
        self.setSubTitle(
            "Pan zero = midpoint of discovered mechanical pan limits (symmetric "
            "mechanism assumption). Tilt zero = wherever you set it with a phone "
            "level now; tilt limits are asymmetric on most differential heads "
            "and that's fine."
        )
        lay = QVBoxLayout(self)

        self.info = QLabel("Click 'Compute' to set pan zero from limit midpoint.")
        lay.addWidget(self.info)
        btn = QPushButton("Compute pan zero from limits")
        btn.clicked.connect(self._compute_pan_zero)
        lay.addWidget(btn)

        lay.addSpacing(10)
        tilt_box = QFrame()
        tilt_box.setFrameShape(QFrame.Box)
        tilt_lay = QVBoxLayout(tilt_box)
        tilt_lay.addWidget(QLabel(
            "Hand-position (or jog) the head so it is exactly level with a phone "
            "level, then click 'Set tilt zero here'. Torque is on — you can jog, "
            "or temporarily disable torque to hand-hold."
        ))
        tilt_row = QHBoxLayout()
        self.torque_toggle = QCheckBox("Torque ON")
        self.torque_toggle.setChecked(True)
        self.torque_toggle.stateChanged.connect(self._toggle_torque)
        tilt_row.addWidget(self.torque_toggle)
        jog_up = QPushButton("Tilt +0.5°")
        jog_dn = QPushButton("Tilt −0.5°")
        jog_up.clicked.connect(lambda: self._tilt_nudge(+0.5))
        jog_dn.clicked.connect(lambda: self._tilt_nudge(-0.5))
        tilt_row.addWidget(jog_up)
        tilt_row.addWidget(jog_dn)
        tilt_set = QPushButton("Set tilt zero HERE")
        tilt_set.clicked.connect(self._set_tilt_zero)
        tilt_row.addWidget(tilt_set)
        tilt_lay.addLayout(tilt_row)
        lay.addWidget(tilt_box)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        lay.addStretch(1)

        self._pan_zero_done = False
        self._tilt_zero_done = False

    def initializePage(self) -> None:
        ctrl = self.state.controller
        if ctrl is not None:
            try:
                ctrl.enable_torque(True)
                ctrl.set_head_pose(0.0, 0.0)
            except Exception as e:
                log.warning(f"refine init: {e}")

    def _compute_pan_zero(self) -> None:
        """Compute pan midpoint in tick-space, re-origin motor zeros to put
        the midpoint at pan=0°. Also update pan_limits_deg symmetrically
        (take the tighter of the two half-ranges)."""
        s = self.state
        cfg = s.cfg
        needed = [
            s.pan_min_raw_l, s.pan_min_raw_r, s.pan_max_raw_l, s.pan_max_raw_r
        ]
        if any(v is None for v in needed):
            self.info.setText("<span style='color:#e66'>Pan limits not fully captured — go back to step 5.</span>")
            return

        # Midpoint ticks
        mid_l = (s.pan_min_raw_l + s.pan_max_raw_l) // 2
        mid_r = (s.pan_min_raw_r + s.pan_max_raw_r) // 2
        cfg.left_zero_tick = mid_l
        cfg.right_zero_tick = mid_r

        # Compute what the raw pan limits work out to in head degrees
        # with the new zero (use the updated cfg via forward kinematics)
        from companion.motor.kinematics import ticks_to_head_pose
        pan_lo, _ = ticks_to_head_pose(s.pan_min_raw_l, s.pan_min_raw_r, cfg)
        pan_hi, _ = ticks_to_head_pose(s.pan_max_raw_l, s.pan_max_raw_r, cfg)
        # These should be approximately symmetric around 0 now
        span = min(abs(pan_lo), abs(pan_hi)) - 2.0                # 2° safety margin
        span = max(5.0, min(90.0, span))
        cfg.pan_limits_deg = [-span, span]

        self.info.setText(
            f"Pan center set to midpoint (L_zero={mid_l}, R_zero={mid_r}). "
            f"Effective pan limits after 2° margin: ±{span:.1f}°."
        )
        self._pan_zero_done = True
        self._update_summary()
        self.completeChanged.emit()

    def _tilt_nudge(self, deg: float) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            current_pan = ctrl.state.pan_deg
            current_tilt = ctrl.state.tilt_deg
            ctrl.set_head_pose(current_pan, current_tilt + deg)
        except Exception as e:
            log.warning(f"tilt nudge: {e}")

    def _toggle_torque(self, checked: int) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        try:
            ctrl.enable_torque(bool(checked))
        except Exception as e:
            log.warning(f"torque toggle: {e}")

    def _set_tilt_zero(self) -> None:
        """Take the *current* encoder positions as the tilt-zero reference —
        but only in the tilt axis (pan zero was just set). We shift both
        motor zeros by the common-mode (sum) of the current tick delta."""
        ctrl = self.state.controller
        if ctrl is None:
            return
        cfg = self.state.cfg
        try:
            lt, rt = ctrl.read_raw_ticks()
        except Exception as e:
            log.warning(f"tilt zero read: {e}")
            return
        # Common-mode (tilt-axis) delta in ticks: we want current tilt reading = 0.
        # Shift both zeros by half the common-mode to absorb it, preserving pan.
        cm = ((lt - cfg.left_zero_tick) * cfg.left_direction
              + (rt - cfg.right_zero_tick) * cfg.right_direction) / 2
        # Back out the per-motor adjustment that zeros tilt without changing pan.
        delta = int(round(cm))
        cfg.left_zero_tick += delta * cfg.left_direction
        cfg.right_zero_tick += delta * cfg.right_direction

        # Recompute tilt limits relative to the new zero
        s = self.state
        if all(v is not None for v in [s.tilt_min_raw_l, s.tilt_min_raw_r,
                                        s.tilt_max_raw_l, s.tilt_max_raw_r]):
            from companion.motor.kinematics import ticks_to_head_pose
            _, tilt_lo = ticks_to_head_pose(s.tilt_min_raw_l, s.tilt_min_raw_r, cfg)
            _, tilt_hi = ticks_to_head_pose(s.tilt_max_raw_l, s.tilt_max_raw_r, cfg)
            lo, hi = sorted([tilt_lo, tilt_hi])
            # apply 2° safety margin inward, asymmetric
            cfg.tilt_limits_deg = [lo + 2.0, hi - 2.0]

        self._tilt_zero_done = True
        self._update_summary()
        self.completeChanged.emit()

    def _update_summary(self) -> None:
        cfg = self.state.cfg
        self.summary.setText(
            f"<pre>"
            f"Left zero  : {cfg.left_zero_tick}\n"
            f"Right zero : {cfg.right_zero_tick}\n"
            f"Pan limits : {cfg.pan_limits_deg[0]:+.1f}° .. {cfg.pan_limits_deg[1]:+.1f}°\n"
            f"Tilt limits: {cfg.tilt_limits_deg[0]:+.1f}° .. {cfg.tilt_limits_deg[1]:+.1f}°"
            f"</pre>"
        )

    def isComplete(self) -> bool:
        return self._pan_zero_done and self._tilt_zero_done


# ── Page 7: Gear ratio measurement ──────────────────────────────────────────

class GearRatioPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("7. Gear ratio measurement (empirical)")
        self.setSubTitle(
            "Commands both motors a known amount in the same direction (pure pitch). "
            "Use a phone level to measure the *actual* head pitch change, enter it, "
            "and click Solve — we'll update gear_ratio_measured."
        )
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Hold the robot steady. Click 'Command +500 ticks both motors' — head will tilt. "
            "Measure head pitch change with a phone level."
        ))
        row = QHBoxLayout()
        self.cmd_btn = QPushButton("Command +500 ticks both (same direction)")
        self.cmd_btn.clicked.connect(self._command_move)
        row.addWidget(self.cmd_btn)
        lay.addLayout(row)

        form = QFormLayout()
        self.measured_deg_spin = QDoubleSpinBox()
        self.measured_deg_spin.setRange(-180, 180)
        self.measured_deg_spin.setDecimals(2)
        self.measured_deg_spin.setValue(0.0)
        form.addRow("Measured head pitch change (°):", self.measured_deg_spin)
        self.solve_btn = QPushButton("Solve gear ratio")
        self.solve_btn.clicked.connect(self._solve)
        form.addRow(self.solve_btn)
        lay.addLayout(form)

        self.result_label = QLabel("Nominal gear ratio: 2.0 (from CAD 40/20).")
        self.result_label.setWordWrap(True)
        lay.addWidget(self.result_label)

        skip = QCheckBox("Skip — keep nominal gear ratio")
        skip.toggled.connect(self._on_skip_toggled)
        lay.addWidget(skip)

        lay.addStretch(1)
        self._solved = False
        self._skipped = False
        self._commanded_motor_deg = 0.0

    def initializePage(self) -> None:
        cfg = self.state.cfg
        self.result_label.setText(
            f"Nominal gear ratio: {cfg.gear_ratio_nominal:.3f}. "
            f"Current measured: {cfg.gear_ratio_measured:.3f}."
        )

    def _command_move(self) -> None:
        """Raw +500 ticks on both motors in the same direction
        (after direction sign correction). That's a pure 'theta_L == theta_R'
        input which is pure pitch."""
        ctrl = self.state.controller
        if ctrl is None:
            return
        cfg = self.state.cfg
        try:
            lt, rt = ctrl.read_raw_ticks()
        except Exception as e:
            log.warning(f"gear-meas read: {e}")
            return
        delta = 500
        new_l = max(0, min(4095, lt + delta * cfg.left_direction))
        new_r = max(0, min(4095, rt + delta * cfg.right_direction))
        try:
            ctrl.write_raw_ticks(new_l, new_r)
        except Exception as e:
            log.warning(f"gear-meas write: {e}")
            return
        # Motor shaft delta (corrected for per-motor sign) in degrees — equal for both
        self._commanded_motor_deg = ticks_to_degrees(delta)

    def _solve(self) -> None:
        if self._commanded_motor_deg == 0.0:
            self.result_label.setText("<span style='color:#e66'>Run the command step first.</span>")
            return
        measured = self.measured_deg_spin.value()
        if abs(measured) < 0.01:
            self.result_label.setText("<span style='color:#e66'>Measured angle ~0; enter a real value.</span>")
            return
        # pitch = (θ_L + θ_R) / (2 · gear_ratio); with equal motor deltas:
        # pitch = θ_motor / gear_ratio  →  gear_ratio = θ_motor / pitch
        g = abs(self._commanded_motor_deg / measured)
        self.state.cfg.gear_ratio_measured = g
        self.result_label.setText(
            f"Commanded {self._commanded_motor_deg:+.2f}° per motor shaft; "
            f"measured {measured:+.2f}° of head pitch. "
            f"→ gear_ratio_measured = {g:.3f}"
        )
        self._solved = True
        self.completeChanged.emit()

    def _on_skip_toggled(self, on: bool) -> None:
        self._skipped = on
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._solved or self._skipped


# ── Page 8: Verify ──────────────────────────────────────────────────────────

class VerifyPage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("8. Verify with live preview")
        self.setSubTitle(
            "Drag the sliders, watch the 3D head (left) follow. Sweep to the "
            "soft-limit edges to confirm nothing binds. If it looks right, "
            "click Next to save."
        )
        main = QHBoxLayout(self)

        controls = QVBoxLayout()
        self.pan_slider = QSlider(Qt.Horizontal)
        self.pan_slider.setMinimum(int(state.cfg.pan_limits_deg[0] * 10))
        self.pan_slider.setMaximum(int(state.cfg.pan_limits_deg[1] * 10))
        self.pan_slider.setValue(0)
        self.pan_value = QLabel("Pan: 0.0°")
        # Update the label instantly during drag, but throttle motor commands
        # via _send_timer so the servos aren't thrashed by every pixel of
        # drag (each new goal pre-empts the previous and creates the
        # cross-coupling tilt-jitter you saw).
        self.pan_slider.valueChanged.connect(self._on_changed)
        self.pan_slider.sliderReleased.connect(self._send_now)
        controls.addWidget(self.pan_value)
        controls.addWidget(self.pan_slider)

        self.tilt_slider = QSlider(Qt.Horizontal)
        self.tilt_slider.setMinimum(int(state.cfg.tilt_limits_deg[0] * 10))
        self.tilt_slider.setMaximum(int(state.cfg.tilt_limits_deg[1] * 10))
        self.tilt_slider.setValue(0)
        self.tilt_value = QLabel("Tilt: 0.0°")
        self.tilt_slider.valueChanged.connect(self._on_changed)
        self.tilt_slider.sliderReleased.connect(self._send_now)
        controls.addWidget(self.tilt_value)
        controls.addWidget(self.tilt_slider)

        home_btn = QPushButton("Home (0, 0)")
        home_btn.clicked.connect(self._home)
        controls.addWidget(home_btn)
        controls.addStretch(1)
        main.addLayout(controls, stretch=0)

        self.preview = HeadPreviewWidget()
        main.addWidget(self.preview, stretch=1)

        # Throttle: on slider drag, coalesce updates so we send at most one
        # goal per ~80 ms. Final position is guaranteed via sliderReleased.
        self._send_timer = QTimer(self)
        self._send_timer.setSingleShot(True)
        self._send_timer.timeout.connect(self._send_now)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.setInterval(int(1000 / max(1, state.cfg.poll_hz)))

    def initializePage(self) -> None:
        self.preview.set_limits(
            tuple(self.state.cfg.pan_limits_deg),
            tuple(self.state.cfg.tilt_limits_deg),
        )
        self.pan_slider.setMinimum(int(self.state.cfg.pan_limits_deg[0] * 10))
        self.pan_slider.setMaximum(int(self.state.cfg.pan_limits_deg[1] * 10))
        self.tilt_slider.setMinimum(int(self.state.cfg.tilt_limits_deg[0] * 10))
        self.tilt_slider.setMaximum(int(self.state.cfg.tilt_limits_deg[1] * 10))
        # Hook right-drag on the 3D preview to head movement
        try:
            self.preview.head_drag_delta.connect(self._on_drag)
        except Exception:
            pass
        ctrl = self.state.controller
        if ctrl is not None:
            try:
                ctrl.reload_config(self.state.cfg)
                ctrl.enable_torque(True)
                # Faster than jog speed so big slider sweeps complete quickly;
                # 1500 ticks/s = ~370°/s of motor shaft.
                ctrl.bus.set_goal_speed(self.state.cfg.left_servo_id, 1500)
                ctrl.bus.set_goal_speed(self.state.cfg.right_servo_id, 1500)
                # Time-mode: both motors complete each move in the same time
                # window (~150 ms). Eliminates the transient tilt during a
                # pure-pan command (and vice-versa) caused by one motor
                # finishing slightly before the other. 150 chosen as a
                # compromise — small enough to feel responsive, large enough
                # that the longest sweeps still complete within the window.
                ctrl.bus.set_goal_time(self.state.cfg.left_servo_id, 150)
                ctrl.bus.set_goal_time(self.state.cfg.right_servo_id, 150)
                ctrl.set_head_pose(0.0, 0.0)
            except Exception as e:
                log.warning(f"verify init: {e}")
        self._timer.start()

    def cleanupPage(self) -> None:
        self._timer.stop()
        self._send_timer.stop()
        ctrl = self.state.controller
        if ctrl is not None:
            # Disable time-mode so downstream pages use GOAL_SPEED
            try:
                ctrl.bus.set_goal_time(self.state.cfg.left_servo_id, 0)
                ctrl.bus.set_goal_time(self.state.cfg.right_servo_id, 0)
            except Exception:
                pass
        try:
            self.preview.head_drag_delta.disconnect(self._on_drag)
        except Exception:
            pass

    def _on_drag(self, dpan: float, dtilt: float) -> None:
        """Right-click-drag delta from the 3D preview. Update sliders (which
        clamp to limits) and send the new pose."""
        new_pan = (self.pan_slider.value() + dpan * 10)
        new_tilt = (self.tilt_slider.value() + dtilt * 10)
        # setValue clamps to the slider's min/max (= the soft limits)
        self.pan_slider.setValue(int(new_pan))
        self.tilt_slider.setValue(int(new_tilt))
        # _on_changed already fired via valueChanged; ensure prompt commit
        if not self._send_timer.isActive():
            self._send_timer.start(40)

    def _on_changed(self) -> None:
        """Slider moved (drag or click): update labels immediately, but
        coalesce servo writes via the throttle timer."""
        pan = self.pan_slider.value() / 10.0
        tilt = self.tilt_slider.value() / 10.0
        self.pan_value.setText(f"Pan: {pan:+.1f}°")
        self.tilt_value.setText(f"Tilt: {tilt:+.1f}°")
        if not self._send_timer.isActive():
            self._send_timer.start(80)   # ~12 Hz max command rate

    def _send_now(self) -> None:
        """Push the current slider values to the servos. Called by the
        throttle timer and by sliderReleased (final commit)."""
        self._send_timer.stop()
        ctrl = self.state.controller
        if ctrl is None:
            return
        pan = self.pan_slider.value() / 10.0
        tilt = self.tilt_slider.value() / 10.0
        try:
            ctrl.set_head_pose(pan, tilt)
        except Exception as e:
            log.warning(f"verify move: {e}")

    def _home(self) -> None:
        self.pan_slider.setValue(0)
        self.tilt_slider.setValue(0)
        self._send_now()

    def _refresh(self) -> None:
        ctrl = self.state.controller
        if ctrl is None:
            return
        self.preview.set_pose(
            ctrl.state.pan_deg, ctrl.state.tilt_deg,
            ctrl.state.target_pan_deg, ctrl.state.target_tilt_deg,
        )
        if ctrl.state.left and ctrl.state.right:
            self.preview.set_telemetry(
                ctrl.state.left.position_tick,
                ctrl.state.right.position_tick,
                ctrl.state.left.temperature_c,
                ctrl.state.right.temperature_c,
            )


# ── Page 9: Save ────────────────────────────────────────────────────────────

class SavePage(QWizardPage):
    def __init__(self, state: WizardState):
        super().__init__()
        self.state = state
        self.setTitle("9. Save to config.yaml")
        self.setSubTitle("Review and write the calibration back to disk.")
        lay = QVBoxLayout(self)
        self.summary_label = QLabel("(summary)")
        font = self.summary_label.font()
        font.setFamily("Monospace")
        self.summary_label.setFont(font)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.summary_label)

        self.save_btn = QPushButton(f"Save to {state.yaml_path}")
        self.save_btn.clicked.connect(self._save)
        lay.addWidget(self.save_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)

        self._saved = False

    def initializePage(self) -> None:
        cfg = self.state.cfg
        result = self._build_result()
        self.summary_label.setText("<pre>" + calibration_summary(result) + "</pre>")

    def _build_result(self) -> CalibrationResult:
        cfg = self.state.cfg
        return CalibrationResult(
            left_zero_tick=cfg.left_zero_tick,
            right_zero_tick=cfg.right_zero_tick,
            left_direction=cfg.left_direction,
            right_direction=cfg.right_direction,
            gear_ratio_measured=cfg.gear_ratio_measured,
            invert_pan=cfg.invert_pan,
            invert_tilt=cfg.invert_tilt,
            pan_limits_deg=list(cfg.pan_limits_deg),
            tilt_limits_deg=list(cfg.tilt_limits_deg),
            backlash_deg=cfg.backlash_deg,
            left_servo_id=cfg.left_servo_id,
            right_servo_id=cfg.right_servo_id,
        )

    def _save(self) -> None:
        try:
            save_to_config_yaml(self._build_result(), self.state.yaml_path)
        except Exception as e:
            self.status.setText(f"<span style='color:#e66'>Save failed: {e}</span>")
            return
        self.status.setText(
            f"<span style='color:#4a4'>Saved to {self.state.yaml_path}.</span>"
        )
        self._saved = True
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._saved
