"""Motor module CLI — connect, scan, calibrate, simulate, one-shot test.

Usage:
    python -m companion.motor.cli scan
    python -m companion.motor.cli set-single-turn          # fixes multi-turn spin
    python -m companion.motor.cli assign-id --from 1 --to 2 [--port /dev/ttyUSB0]
    python -m companion.motor.cli calibrate [--sim]
    python -m companion.motor.cli sim
    python -m companion.motor.cli test --pan 20 --tilt 5 [--sim]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

from companion.core.config import MotorConfig, load_config

log = logging.getLogger(__name__)


def _config_and_path(use_sim: bool = False) -> tuple[MotorConfig, str]:
    """Load motor config from config.yaml. CLI forces real hardware by default;
    pass use_sim=True (via --sim flag) to explicitly use the simulator bus,
    regardless of what `motor.sim_only` says in config.yaml.
    """
    project_root = Path(__file__).resolve().parents[2]
    yaml_path = str(project_root / "config.yaml")
    app_cfg = load_config(yaml_path)
    motor_cfg = copy.deepcopy(app_cfg.motor)
    motor_cfg.sim_only = bool(use_sim)
    return motor_cfg, yaml_path


def cmd_scan(args) -> int:
    cfg, _ = _config_and_path(use_sim=args.sim)
    if args.port:
        cfg.port = args.port
    from companion.motor.controller import HeadController

    ctrl = HeadController(cfg)
    try:
        ctrl.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1
    try:
        found = ctrl.bus.scan(range(1, 20))
        print(f"Servos found on {'SIM' if cfg.sim_only else cfg.port} @ {cfg.baudrate} baud: {found}")
        for sid in found:
            tel = ctrl.bus.read_telemetry(sid)
            print(
                f"  id={sid:3d} pos={tel.position_tick:4d}  "
                f"temp={tel.temperature_c:.1f}°C  volt={tel.voltage_v:.1f}V"
            )
    finally:
        ctrl.disconnect()
    return 0


def cmd_diag(args) -> int:
    """Diagnostic: try all common ST3215 baud rates and a wide ID range.
    Useful when scan at the configured baud rate finds nothing."""
    cfg, _ = _config_and_path(use_sim=False)
    if args.port:
        cfg.port = args.port
    from companion.motor.bus import ST3215Bus

    baud_rates = [1000000, 500000, 250000, 115200, 57600, 38400]
    id_range = range(1, 254)
    print(f"Diagnostic scan on {cfg.port} — trying {len(baud_rates)} baud rates, IDs 1..253")
    print("(Ctrl-C to abort; a full sweep takes ~30 seconds per baud rate)")
    any_found = False
    for baud in baud_rates:
        print(f"\n— {baud} baud:")
        bus = ST3215Bus(port=cfg.port, baudrate=baud)
        try:
            bus.open()
        except Exception as e:
            print(f"  open failed: {e}")
            continue
        try:
            found = []
            for sid in id_range:
                if bus.ping(sid):
                    found.append(sid)
            if found:
                any_found = True
                print(f"  ✓ Found IDs: {found}")
            else:
                print(f"  (no response)")
        finally:
            bus.close()
    if not any_found:
        print(
            "\nNo servos responded at any baud rate. Most likely:\n"
            "  1. Servo power is OFF (ST3215 needs 7.4V–12V external supply; "
            "USB alone doesn't power them)\n"
            "  2. Data wire not connected, or TX/RX swapped\n"
            "  3. Wrong serial adapter selected\n"
            "Check servo LEDs — powered ST3215s usually blink briefly on power-up."
        )
    return 0


def cmd_dump_registers(args) -> int:
    """Read back servo EEPROM/RAM settings relevant to position control.
    Helps verify whether set-single-turn actually stuck."""
    cfg, _ = _config_and_path(use_sim=False)
    if args.port:
        cfg.port = args.port
    from companion.motor.bus import ST3215Bus

    bus = ST3215Bus(port=cfg.port, baudrate=cfg.baudrate)
    bus.open()
    try:
        found = bus.scan(range(1, 20))
        if not found:
            print("No servos found.")
            return 1
        for sid in found:
            print(f"\n── Servo id={sid} ─────────────────────────")
            reads = [
                ("MODEL_NUMBER (reg 3, 2B)", 3, 2),
                ("FW_VERSION_MAIN (reg 0, 1B)", 0, 1),
                ("FW_VERSION_SUB (reg 1, 1B)", 1, 1),
                ("MIN_ANGLE_LIMIT (reg 9, 2B)", 9, 2),
                ("MAX_ANGLE_LIMIT (reg 11, 2B)", 11, 2),
                ("MAX_TEMPERATURE (reg 13, 1B, °C)", 13, 1),
                ("MAX_INPUT_VOLTAGE (reg 14, 1B, /10=V)", 14, 1),
                ("MIN_INPUT_VOLTAGE (reg 15, 1B, /10=V)", 15, 1),
                ("MAX_TORQUE (reg 16, 2B)", 16, 2),
                ("PHASE (reg 18, 1B)", 18, 1),
                ("UNLOAD_CONDITION (reg 19, 1B)", 19, 1),
                ("LED_ALARM (reg 20, 1B)", 20, 1),
                ("CW_INSENSITIVE (reg 26, 1B)", 26, 1),
                ("CCW_INSENSITIVE (reg 27, 1B)", 27, 1),
                ("ANGULAR_RESOLUTION (reg 30, 1B)", 30, 1),
                ("OFFSET (reg 31, 2B signed)", 31, 2),
                ("MODE (reg 33, 1B)", 33, 1),
                ("TORQUE_ENABLE (reg 40, 1B)", 40, 1),
                ("GOAL_ACCELERATION (reg 41, 1B)", 41, 1),
                ("GOAL_POSITION (reg 42, 2B)", 42, 2),
                ("GOAL_SPEED (reg 46, 2B)", 46, 2),
                ("LOCK (reg 55, 1B)", 55, 1),
                ("PRESENT_POSITION (reg 56, 2B)", 56, 2),
                ("PRESENT_VOLTAGE (reg 62, 1B, /10=V)", 62, 1),
            ]
            for name, reg, width in reads:
                read = (
                    bus._packet.read1ByteTxRx if width == 1
                    else bus._packet.read2ByteTxRx
                )
                val, result, error = read(bus._port, sid, reg)
                if result != 0:
                    print(f"  {name:40s} = <read failed code={result}>")
                else:
                    # Sign-interpret OFFSET (sign bit is 0x8000 for Feetech STS)
                    if reg == 31 and (val & 0x8000):
                        signed = -(val & 0x7FFF)
                        print(f"  {name:40s} = {val} (signed: {signed})")
                    else:
                        print(f"  {name:40s} = {val}")
            print(
                "\n  Expected for single-turn bounded mode:\n"
                "    MIN_ANGLE_LIMIT = 0, MAX_ANGLE_LIMIT = 4095, MODE = 0"
            )
    finally:
        bus.close()
    return 0


def cmd_factory_reset(args) -> int:
    """Factory-reset each servo's EEPROM to defaults via Feetech instruction
    0x06. Keeps ID and baud rate by default — pass --full to reset those too
    (servo will revert to ID 1 at the default baud).

    Use this when targeted register writes (set-single-turn, reset-safety-
    limits, recenter) can't recover the servo to sensible behavior."""
    cfg, _ = _config_and_path(use_sim=False)
    if args.port:
        cfg.port = args.port
    from companion.motor.bus import ST3215Bus

    bus = ST3215Bus(port=cfg.port, baudrate=cfg.baudrate)
    bus.open()
    try:
        found = bus.scan(range(1, 20))
        if not found:
            print("No servos found.")
            return 1
        print(f"Found servos: {found}")
        keep_id = not args.full
        for sid in found:
            try:
                bus.factory_reset(sid, keep_id=keep_id)
                print(f"  id={sid}: factory reset issued (keep_id={keep_id}).")
            except Exception as e:
                print(f"  id={sid}: FAILED — {e}")
        print(
            "\nDone. Power-cycle the servos now. Then re-run the bring-up flow:\n"
            "  1. scan       — confirm IDs\n"
            "  2. set-single-turn + power-cycle\n"
            "  3. reset-safety-limits + power-cycle  (optional, defaults are usually fine)\n"
            "  4. recenter (hand-hold head at center) + power-cycle\n"
            "  5. dump-registers — verify everything sane\n"
            "  6. jog-test — verify small moves\n"
            "  7. calibrate"
        )
    finally:
        bus.close()
    return 0


def cmd_reset_safety_limits(args) -> int:
    """Restore sensible voltage/temperature/torque thresholds in the servo
    EEPROM. Use when dump-registers shows MAX_INPUT_VOLTAGE at a weird low
    value (like 80) — that causes a persistent 0x01 overvoltage alarm."""
    cfg, _ = _config_and_path(use_sim=False)
    if args.port:
        cfg.port = args.port
    from companion.motor.bus import ST3215Bus

    bus = ST3215Bus(port=cfg.port, baudrate=cfg.baudrate)
    bus.open()
    try:
        found = bus.scan(range(1, 20))
        if not found:
            print("No servos found.")
            return 1
        print(f"Found servos: {found}")
        for sid in found:
            try:
                bus.reset_safety_limits(sid)
                print(f"  id={sid}: safety limits restored to defaults.")
            except Exception as e:
                print(f"  id={sid}: FAILED — {e}")
        print(
            "\nDone. Power-cycle the servos now. Then run dump-registers to\n"
            "confirm MAX_INPUT_VOLTAGE = 140 and the 0x01 alarm should clear."
        )
    finally:
        bus.close()
    return 0


def cmd_jog_test(args) -> int:
    """Do a series of tiny relative moves on one servo and report the actual
    before/after positions. Diagnostic for 'motor spins a lot on a small
    command' — lets us confirm at the bus level (no wizard) whether the servo
    is obeying the goal we wrote."""
    import time
    cfg, _ = _config_and_path(use_sim=False)
    if args.port:
        cfg.port = args.port
    from companion.motor.bus import ST3215Bus

    bus = ST3215Bus(port=cfg.port, baudrate=cfg.baudrate)
    bus.open()
    try:
        sid = args.id
        delta = args.delta
        # Read present and pin goal there BEFORE enabling torque
        def read_goal_back(sid):
            val, r, _ = bus._packet.read2ByteTxRx(bus._port, sid, 42)
            return val if r == 0 else -1

        def read_goal_speed(sid):
            val, r, _ = bus._packet.read2ByteTxRx(bus._port, sid, 46)
            return val if r == 0 else -1

        start = bus.read_position(sid)
        gs_before = read_goal_speed(sid)
        # After power cycle, GOAL_SPEED resets to 0 (= unlimited speed) and
        # the motor will fling itself at goals. Pin a sane speed and accel
        # before doing any motion test.
        bus.set_goal_speed(sid, 500)
        bus.set_goal_acceleration(sid, 50)
        gs_after = read_goal_speed(sid)
        print(f"Initial PRESENT_POSITION id={sid}: {start}")
        print(f"GOAL_SPEED before/after init: {gs_before} → {gs_after}")
        bus.write_goal(sid, start % 4096)
        readback = read_goal_back(sid)
        print(f"Wrote goal={start % 4096}; readback={readback} "
              f"{'✓' if readback == start % 4096 else '✗ MISMATCH'}")
        bus.enable_torque(sid, True)
        time.sleep(0.3)
        after_hold = bus.read_position(sid)
        print(f"After torque-on (goal=current, wait 300ms): {after_hold}  "
              f"(drift: {after_hold - start:+d})")
        target = start % 4096
        for i in range(args.n):
            target = max(0, min(4095, target + delta))
            bus.write_goal(sid, target)
            readback = read_goal_back(sid)
            before = bus.read_position(sid)
            time.sleep(0.5)
            after = bus.read_position(sid)
            print(f"  move {i+1}: wrote_goal={target:4d} readback={readback:5d}  "
                  f"before={before:5d}  after={after:5d}  "
                  f"actual_delta={after - before:+5d}  (expected ~{delta:+d})")
        print("\nDisabling torque.")
        bus.enable_torque(sid, False)
    finally:
        bus.close()
    return 0


def cmd_recenter(args) -> int:
    """Reset each servo's internal encoder so its current mechanical position
    becomes tick 2048. Use when dump-registers shows PRESENT_POSITION out of
    the 0..4095 range despite single-turn limits being set correctly.

    Flow: disable torque → prompt user to hand-position head → write calibrate.
    """
    cfg, _ = _config_and_path(use_sim=False)
    if args.port:
        cfg.port = args.port
    from companion.motor.bus import ST3215Bus

    bus = ST3215Bus(port=cfg.port, baudrate=cfg.baudrate)
    bus.open()
    try:
        found = bus.scan(range(1, 20))
        if not found:
            print("No servos found.")
            return 1
        print(f"Found servos: {found}")
        for sid in found:
            try:
                bus.enable_torque(sid, False)
            except Exception as e:
                print(f"  id={sid}: disable torque failed — {e}")
        print(
            "\nTorque is OFF. Hand-position the head so it looks forward and level.\n"
            "Hold it steady — the position at the moment you press ENTER becomes\n"
            "the new tick=2048 (mechanical center).\n"
        )
        try:
            input("Press ENTER to capture current position as center... ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        for sid in found:
            try:
                bus.recenter(sid)
                print(f"  id={sid}: recentered.")
            except Exception as e:
                print(f"  id={sid}: FAILED — {e}")
        print(
            "\nDone. Power-cycle the servos now. Then run:\n"
            "  python -m companion.motor.cli dump-registers\n"
            "PRESENT_POSITION should read ~2048 on both."
        )
    finally:
        bus.close()
    return 0


def cmd_set_single_turn(args) -> int:
    """Force every servo found on the bus into single-turn position mode.
    Use when `scan` reports positions outside 0..4095 (multi-turn / wheel mode)."""
    cfg, _ = _config_and_path(use_sim=False)
    if args.port:
        cfg.port = args.port
    from companion.motor.bus import ST3215Bus

    bus = ST3215Bus(port=cfg.port, baudrate=cfg.baudrate)
    bus.open()
    try:
        found = bus.scan(range(1, 20))
        if not found:
            print("No servos found on the bus.")
            return 1
        print(f"Found servos: {found}")
        for sid in found:
            try:
                bus.configure_single_turn(sid)
                print(f"  id={sid}: set to single-turn mode (0..4095).")
            except Exception as e:
                print(f"  id={sid}: FAILED — {e}")
        print(
            "\nDone. Power-cycle the servos now (pull & replug power) to clear any\n"
            "latched multi-turn count. Then re-run `scan` — positions should read 0..4095."
        )
    finally:
        bus.close()
    return 0


def cmd_assign_id(args) -> int:
    cfg, _ = _config_and_path(use_sim=args.sim)
    if args.port:
        cfg.port = args.port
    from companion.motor.controller import HeadController

    ctrl = HeadController(cfg)
    ctrl.connect()
    try:
        if not ctrl.bus.ping(args.from_id):
            print(f"No servo found at id {args.from_id}")
            return 1
        ctrl.bus.set_id(args.from_id, args.to_id)
        print(f"Servo id {args.from_id} → {args.to_id}")
        if ctrl.bus.ping(args.to_id):
            print(f"Confirmed: servo now responds at id {args.to_id}")
        else:
            print(f"Warning: servo did not respond at id {args.to_id} after assignment")
    finally:
        ctrl.disconnect()
    return 0


def cmd_calibrate(args) -> int:
    cfg, yaml_path = _config_and_path(use_sim=args.sim)
    if args.port:
        cfg.port = args.port
    from PyQt5.QtWidgets import QApplication
    from companion.ui.calibration_window import CalibrationWizard

    app = QApplication.instance() or QApplication(sys.argv)
    wiz = CalibrationWizard(cfg, yaml_path)
    wiz.show()
    return app.exec_()


def cmd_sim(args) -> int:
    """Standalone simulator — sliders + live 3D preview, no wizard."""
    cfg, _ = _config_and_path(use_sim=True)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
    )
    from companion.motor.controller import HeadController
    from companion.ui.widgets.head_preview import HeadPreviewWidget

    app = QApplication.instance() or QApplication(sys.argv)
    ctrl = HeadController(cfg)
    ctrl.connect()
    ctrl.enable_torque(True)

    win = QWidget()
    win.setWindowTitle("Head Simulator (no hardware)")
    win.resize(900, 600)
    main = QHBoxLayout(win)

    controls = QVBoxLayout()
    pan_lbl = QLabel("Pan: 0.0°")
    pan = QSlider(Qt.Horizontal)
    pan.setMinimum(int(cfg.pan_limits_deg[0] * 10))
    pan.setMaximum(int(cfg.pan_limits_deg[1] * 10))
    tilt_lbl = QLabel("Tilt: 0.0°")
    tilt = QSlider(Qt.Horizontal)
    tilt.setMinimum(int(cfg.tilt_limits_deg[0] * 10))
    tilt.setMaximum(int(cfg.tilt_limits_deg[1] * 10))
    home = QPushButton("Home (0, 0)")
    controls.addWidget(pan_lbl)
    controls.addWidget(pan)
    controls.addWidget(tilt_lbl)
    controls.addWidget(tilt)
    controls.addWidget(home)
    controls.addStretch(1)
    main.addLayout(controls, stretch=0)

    preview = HeadPreviewWidget()
    preview.set_limits(tuple(cfg.pan_limits_deg), tuple(cfg.tilt_limits_deg))
    main.addWidget(preview, stretch=1)

    def on_changed():
        p = pan.value() / 10.0
        t = tilt.value() / 10.0
        pan_lbl.setText(f"Pan: {p:+.1f}°")
        tilt_lbl.setText(f"Tilt: {t:+.1f}°")
        ctrl.set_head_pose(p, t)

    pan.valueChanged.connect(on_changed)
    tilt.valueChanged.connect(on_changed)
    home.clicked.connect(lambda: (pan.setValue(0), tilt.setValue(0)))

    timer = QTimer()
    timer.timeout.connect(
        lambda: preview.set_pose(
            ctrl.state.pan_deg, ctrl.state.tilt_deg,
            ctrl.state.target_pan_deg, ctrl.state.target_tilt_deg,
        )
    )
    timer.start(int(1000 / max(1, cfg.poll_hz)))

    win.show()
    rc = app.exec_()
    ctrl.disconnect()
    return rc


def cmd_raw(args) -> int:
    """Raw-byte test — sends a ping packet with pyserial directly, bypassing
    the scservo_sdk. Tells us whether the adapter is even transmitting, and
    whether the servo is replying at all.
    """
    import serial
    import time
    cfg, _ = _config_and_path(use_sim=False)
    port = args.port or cfg.port
    sid = args.id
    # ST3215 ping packet: FF FF <id> 02 01 <checksum>
    # checksum = ~(id + length + instruction) & 0xFF
    checksum = (~(sid + 0x02 + 0x01)) & 0xFF
    packet = bytes([0xFF, 0xFF, sid, 0x02, 0x01, checksum])
    print(f"Opening {port} @ {cfg.baudrate} baud")
    s = serial.Serial(port, cfg.baudrate, timeout=0.2)
    # Drain any stale bytes
    s.reset_input_buffer()
    print(f"Sending ping for ID {sid}: {packet.hex(' ')}")
    s.write(packet)
    s.flush()
    time.sleep(0.05)
    reply = s.read(16)
    s.close()
    print(f"Received {len(reply)} bytes: {reply.hex(' ') if reply else '(nothing)'}")
    if not reply:
        print("→ No bytes at all. Adapter is not transmitting, OR servo is not powered,")
        print("  OR the signal line isn't connected.")
    elif reply == packet:
        print("→ Got back EXACTLY what we sent — half-duplex echo with NO servo response.")
        print("  The servo didn't hear us, or isn't replying. Check signal wiring / IDs / baud.")
    elif reply.startswith(packet):
        tail = reply[len(packet):]
        print(f"→ Echo + {len(tail)} trailing bytes: {tail.hex(' ')}")
        print("  The trailing bytes should be the servo's reply (expect: FF FF <id> ...).")
    else:
        print("→ Got a reply that doesn't look like pure echo — inspect above.")
    return 0


def cmd_test(args) -> int:
    """One-shot: command a pose and print readback."""
    cfg, _ = _config_and_path(use_sim=args.sim)
    if args.port:
        cfg.port = args.port
    from companion.motor.controller import HeadController

    ctrl = HeadController(cfg)
    ctrl.connect()
    try:
        ctrl.enable_torque(True)
        lt, rt = ctrl.set_head_pose(args.pan, args.tilt)
        print(f"commanded: pan={args.pan:+.1f}°  tilt={args.tilt:+.1f}°")
        print(f"  → L_tick={lt}, R_tick={rt}")
        time.sleep(args.wait)
        pan, tilt = ctrl.get_head_pose()
        print(f"readback : pan={pan:+.2f}°  tilt={tilt:+.2f}°")
    finally:
        ctrl.disconnect()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="companion.motor.cli", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Scan the bus for ST3215 servos")
    p_scan.add_argument("--sim", action="store_true")
    p_scan.add_argument("--port", type=str, default="")
    p_scan.set_defaults(func=cmd_scan)

    p_dump = sub.add_parser(
        "dump-registers",
        help="Read back servo EEPROM/RAM registers (mode, angle limits, position)",
    )
    p_dump.add_argument("--port", type=str, default="")
    p_dump.set_defaults(func=cmd_dump_registers)

    p_sst = sub.add_parser(
        "set-single-turn",
        help="Force every connected servo into single-turn mode (fixes multi-turn spin)",
    )
    p_sst.add_argument("--port", type=str, default="")
    p_sst.set_defaults(func=cmd_set_single_turn)

    p_jt = sub.add_parser(
        "jog-test",
        help="Diagnostic — do N small moves on one servo and print before/after ticks",
    )
    p_jt.add_argument("--id", type=int, default=1)
    p_jt.add_argument("--delta", type=int, default=57, help="ticks per move (57 ≈ 5°)")
    p_jt.add_argument("--n", type=int, default=5)
    p_jt.add_argument("--port", type=str, default="")
    p_jt.set_defaults(func=cmd_jog_test)

    p_rec = sub.add_parser(
        "recenter",
        help="Re-zero each servo so its current physical position = tick 2048",
    )
    p_rec.add_argument("--port", type=str, default="")
    p_rec.set_defaults(func=cmd_recenter)

    p_rsl = sub.add_parser(
        "reset-safety-limits",
        help="Restore default voltage/temp/torque thresholds (clears 0x01 alarm)",
    )
    p_rsl.add_argument("--port", type=str, default="")
    p_rsl.set_defaults(func=cmd_reset_safety_limits)

    p_fr = sub.add_parser(
        "factory-reset",
        help="Send Feetech FACTORY_RESET instruction; keeps ID by default",
    )
    p_fr.add_argument("--port", type=str, default="")
    p_fr.add_argument("--full", action="store_true",
                      help="Also reset ID and baud rate (default: keep them)")
    p_fr.set_defaults(func=cmd_factory_reset)

    p_assign = sub.add_parser("assign-id", help="Change a servo's ID")
    p_assign.add_argument("--from", dest="from_id", type=int, required=True)
    p_assign.add_argument("--to", dest="to_id", type=int, required=True)
    p_assign.add_argument("--sim", action="store_true")
    p_assign.add_argument("--port", type=str, default="")
    p_assign.set_defaults(func=cmd_assign_id)

    p_cal = sub.add_parser("calibrate", help="Launch the calibration wizard")
    p_cal.add_argument("--sim", action="store_true", help="Simulator-only mode")
    p_cal.add_argument("--port", type=str, default="", help="Override serial port from config.yaml")
    p_cal.set_defaults(func=cmd_calibrate)

    p_sim = sub.add_parser("sim", help="Standalone simulator (no hardware)")
    p_sim.set_defaults(func=cmd_sim)

    p_diag = sub.add_parser("diag", help="Full diagnostic — sweep baud rates + IDs")
    p_diag.add_argument("--port", type=str, default="", help="Override serial port from config.yaml")
    p_diag.set_defaults(func=cmd_diag)

    p_raw = sub.add_parser("raw", help="Send a raw ping via pyserial, bypassing scservo_sdk")
    p_raw.add_argument("--id", type=int, default=1)
    p_raw.add_argument("--port", type=str, default="")
    p_raw.set_defaults(func=cmd_raw)

    p_test = sub.add_parser("test", help="One-shot pose command")
    p_test.add_argument("--pan", type=float, default=0.0)
    p_test.add_argument("--tilt", type=float, default=0.0)
    p_test.add_argument("--wait", type=float, default=1.0)
    p_test.add_argument("--sim", action="store_true")
    p_test.add_argument("--port", type=str, default="", help="Override serial port from config.yaml")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
