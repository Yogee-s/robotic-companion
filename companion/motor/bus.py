"""Servo bus abstraction — ST3215 hardware and simulator, same interface.

HeadController talks to a ServoBus; swapping ST3215Bus for SimulatedBus is how
the simulator-only mode works (no `if sim:` branches scattered through the
controller or calibration code).

ST3215 register map (from Waveshare/Feetech ST series datasheet):
    0x03 (3)  MODEL_NUMBER_L     2 bytes  (read)
    0x05 (5)  ID                 1 byte   (read/write, then must save to EEPROM)
    0x28 (40) TORQUE_ENABLE      1 byte   (1 = on, 0 = off)
    0x29 (41) GOAL_ACCELERATION  1 byte
    0x2A (42) GOAL_POSITION      2 bytes
    0x2E (46) GOAL_SPEED         2 bytes
    0x30 (48) TORQUE_LIMIT       2 bytes  (0..1000)
    0x38 (56) PRESENT_POSITION   2 bytes
    0x3A (58) PRESENT_SPEED      2 bytes
    0x3C (60) PRESENT_LOAD       2 bytes  (current-proxy)
    0x3E (62) PRESENT_VOLTAGE    1 byte   (0.1 V units)
    0x3F (63) PRESENT_TEMPERATURE 1 byte  (°C)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ── Register addresses ──────────────────────────────────────────────────────
_REG_ID = 5
_REG_MIN_ANGLE_LIMIT = 9      # EEPROM, 2 bytes. Set to 0 for single-turn.
_REG_MAX_ANGLE_LIMIT = 11     # EEPROM, 2 bytes. Set to 4095 for single-turn.
_REG_MODE = 33                # EEPROM, 1 byte. 0 = position mode, 1 = wheel mode.
_REG_LOCK = 55                # RAM, 1 byte. 0 = unlocked, 1 = locked (default).
_REG_TORQUE_ENABLE = 40
_REG_GOAL_ACCEL = 41
_REG_GOAL_POSITION = 42
_REG_GOAL_TIME = 44       # Move-completion-time mode. 0 = use GOAL_SPEED.
_REG_GOAL_SPEED = 46
_REG_TORQUE_LIMIT = 48
_REG_PRESENT_POSITION = 56
_REG_PRESENT_SPEED = 58
_REG_PRESENT_LOAD = 60
_REG_PRESENT_VOLTAGE = 62
_REG_PRESENT_TEMPERATURE = 63


@dataclass
class ServoTelemetry:
    servo_id: int
    position_tick: int
    speed: int
    load: int                  # raw register units (0..1000-ish)
    voltage_v: float
    temperature_c: float
    ok: bool                   # False if the read failed / timeout
    error: str = ""


@runtime_checkable
class ServoBus(Protocol):
    """Interface both hardware and simulator buses implement."""

    def open(self) -> None: ...
    def close(self) -> None: ...
    def ping(self, servo_id: int) -> bool: ...
    def scan(self, id_range: range = range(1, 10)) -> list[int]: ...
    def set_id(self, old_id: int, new_id: int) -> None: ...
    def enable_torque(self, servo_id: int, on: bool) -> None: ...
    def set_torque_limit(self, servo_id: int, limit: int) -> None: ...
    def set_goal_speed(self, servo_id: int, speed: int) -> None: ...
    def set_goal_acceleration(self, servo_id: int, accel: int) -> None: ...
    def read_position(self, servo_id: int) -> int: ...
    def read_telemetry(self, servo_id: int) -> ServoTelemetry: ...
    def write_goal(self, servo_id: int, tick: int) -> None: ...
    def sync_write_goal(self, goals: dict[int, int]) -> None: ...


# ── Hardware bus ────────────────────────────────────────────────────────────

class ST3215Bus:
    """Real ST3215 bus via Waveshare USB→TTL adapter + scservo_sdk."""

    def __init__(self, port: str, baudrate: int = 1000000):
        self.port = port
        self.baudrate = baudrate
        self._port = None
        self._packet = None
        self._lock = threading.Lock()

    def open(self) -> None:
        try:
            from scservo_sdk import PortHandler, PacketHandler
        except ImportError as e:
            raise RuntimeError(
                "scservo_sdk is not installed. Run `pip install feetech-servo-sdk` "
                "(PyPI package name; imports as `scservo_sdk`). For simulator-"
                "only work, pass --sim to the CLI."
            ) from e

        self._port = PortHandler(self.port)
        # protocol_end byte order — per official Feetech SDK examples:
        #   STS / SMS series → 0 (little-endian)
        #   SCS series       → 1 (big-endian)
        # The previous value (1) silently byte-swapped every 2-byte register
        # read/write, which made the servo behave as if MIN/MAX angle limits,
        # GOAL_POSITION and PRESENT_POSITION were all garbage values — symptoms
        # were "multi-turn spinning" and "huge moves on small goal writes."
        self._packet = PacketHandler(0)
        if not self._port.openPort():
            raise RuntimeError(f"Failed to open serial port {self.port}")
        if not self._port.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on {self.port}")
        log.info(f"ST3215 bus open on {self.port} @ {self.baudrate} baud")

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
            self._port = None
            self._packet = None

    def _check_result(self, result: int, error: int, op: str) -> None:
        if result != 0:
            raise RuntimeError(f"ST3215 {op} comm error: code={result}")
        if error != 0:
            log.warning(f"ST3215 {op} servo error byte: 0x{error:02x}")

    def ping(self, servo_id: int) -> bool:
        # Retry a few times — ST3215 on half-duplex buses sometimes corrupts
        # the first packet after a direction flip; clean packets get through
        # on retry. Reset the input buffer each time to discard noise bytes
        # left over from a previous garbled read.
        for _attempt in range(3):
            with self._lock:
                try:
                    self._port.ser.reset_input_buffer()
                except Exception:
                    pass
                _model, result, _error = self._packet.ping(self._port, servo_id)
                if result == 0:
                    return True
        return False

    def scan(self, id_range: range = range(1, 10)) -> list[int]:
        found = []
        for sid in id_range:
            if self.ping(sid):
                found.append(sid)
        return found

    def set_id(self, old_id: int, new_id: int) -> None:
        """Change a servo's ID. Caller is responsible for ensuring only one
        servo with old_id is on the bus (bus collision otherwise).

        ID lives in EEPROM — must unlock (LOCK=0) before the write, otherwise
        only the RAM shadow updates and the ID reverts on power cycle."""
        with self._lock:
            # Unlock EEPROM
            r, _ = self._packet.write1ByteTxRx(self._port, old_id, _REG_LOCK, 0)
            if r != 0:
                raise RuntimeError(f"set_id({old_id}->{new_id}) unlock: code={r}")
            time.sleep(0.03)
            # Write new ID
            result, error = self._packet.write1ByteTxRx(
                self._port, old_id, _REG_ID, new_id
            )
            if result != 0:
                raise RuntimeError(f"set_id({old_id}->{new_id}) write: code={result}")
            if error != 0:
                log.warning(f"set_id({old_id}->{new_id}) servo error 0x{error:02x}")
            time.sleep(0.03)
            # Relock EEPROM — servo now responds to new_id, not old_id
            r, _ = self._packet.write1ByteTxRx(self._port, new_id, _REG_LOCK, 1)
            if r != 0:
                log.warning(f"set_id({old_id}->{new_id}) relock: code={r}")

    def factory_reset(self, servo_id: int, keep_id: bool = True) -> None:
        """Send Feetech FACTORY_RESET instruction (0x06). Resets EEPROM to
        defaults. With keep_id=True, ID and baud rate survive the reset
        (mode byte 0x01); with False, everything resets and the servo
        becomes ID 1 at the default baud (mode byte 0xFF).

        Power-cycle is required after for changes to apply."""
        # scservo_sdk doesn't expose factoryReset directly; build the packet
        # manually. Feetech protocol: FF FF <id> <len=4> <inst=0x06> <mode> <chk>
        with self._lock:
            try:
                self._port.ser.reset_input_buffer()
            except Exception:
                pass
            mode = 0x01 if keep_id else 0xFF
            length = 4
            instruction = 0x06
            checksum = (~(servo_id + length + instruction + mode)) & 0xFF
            packet = bytes([0xFF, 0xFF, servo_id, length, instruction, mode, checksum])
            self._port.ser.write(packet)
            self._port.ser.flush()
            time.sleep(0.5)  # firmware needs a moment to reset
            try:
                self._port.ser.reset_input_buffer()
            except Exception:
                pass

    def reset_safety_limits(self, servo_id: int) -> None:
        """Restore sensible EEPROM safety thresholds that commonly get clobbered
        by bad writes. Specifically:
          - MAX_INPUT_VOLTAGE = 140 (14.0 V, default)
          - MIN_INPUT_VOLTAGE = 40  (4.0 V, default)
          - MAX_TEMPERATURE   = 70  (70 °C, default)
          - MAX_TORQUE        = 1000 (default cap)
        Torque must be off. Power-cycle after for the changes to take effect."""
        try:
            self.enable_torque(servo_id, False)
        except Exception:
            pass
        writes = [
            ("unlock EEPROM", _REG_LOCK, 0, 1),
            ("MAX_INPUT_VOLTAGE = 140", 14, 140, 1),
            ("MIN_INPUT_VOLTAGE = 40", 15, 40, 1),
            ("MAX_TEMPERATURE = 70", 13, 70, 1),
            ("MAX_TORQUE = 1000", 16, 1000, 2),
            ("relock EEPROM", _REG_LOCK, 1, 1),
        ]
        for name, reg, val, width in writes:
            write = (
                self._packet.write1ByteTxRx if width == 1
                else self._packet.write2ByteTxRx
            )
            last_err = 0
            for _attempt in range(4):
                with self._lock:
                    try:
                        self._port.ser.reset_input_buffer()
                    except Exception:
                        pass
                    result, error = write(self._port, servo_id, reg, val)
                if result == 0:
                    if error != 0:
                        log.warning(f"{name} id={servo_id}: servo error 0x{error:02x}")
                    break
                last_err = result
                time.sleep(0.03)
            else:
                raise RuntimeError(f"{name} id={servo_id}: comm code={last_err}")
            time.sleep(0.03)

    def recenter(self, servo_id: int) -> None:
        """Reset the servo's internal encoder reference so the *current*
        mechanical position becomes tick 2048. Clears any multi-turn counter
        accumulation. Feetech-specific trick: write 128 to TORQUE_ENABLE.
        EEPROM-persistent — power-cycle recommended after."""
        with self._lock:
            try:
                self._port.ser.reset_input_buffer()
            except Exception:
                pass
            # Unlock first
            r, _ = self._packet.write1ByteTxRx(self._port, servo_id, _REG_LOCK, 0)
            if r != 0:
                raise RuntimeError(f"recenter id={servo_id}: unlock failed code={r}")
            time.sleep(0.03)
            # 128 = calibrate midpoint (Feetech STS/ST convention)
            r, _ = self._packet.write1ByteTxRx(self._port, servo_id, _REG_TORQUE_ENABLE, 128)
            if r != 0:
                raise RuntimeError(f"recenter id={servo_id}: calibrate write failed code={r}")
            time.sleep(0.05)
            # Relock
            r, _ = self._packet.write1ByteTxRx(self._port, servo_id, _REG_LOCK, 1)
            if r != 0:
                raise RuntimeError(f"recenter id={servo_id}: relock failed code={r}")

    def configure_single_turn(
        self, servo_id: int, min_tick: int = 0, max_tick: int = 4095
    ) -> None:
        """Force a servo into single-turn position mode with [min_tick, max_tick]
        travel. Rewrites EEPROM (MODE=0, angle limits) — power-cycle the servo
        after to clear any latched multi-turn counters. Torque must be off.

        ST3215 EEPROM commits take ~10 ms per register; a quick back-to-back
        write can return before the commit finishes and the next op then gets
        a -6 (corrupt reply). We sleep briefly between writes and retry."""
        min_tick = max(0, min(4095, int(min_tick)))
        max_tick = max(0, min(4095, int(max_tick)))
        # Best-effort torque off — some firmware rejects EEPROM writes otherwise
        try:
            self.enable_torque(servo_id, False)
        except Exception:
            pass
        steps = [
            ("unlock EEPROM", _REG_LOCK, 0, 1),
            ("set MODE=0 (position)", _REG_MODE, 0, 1),
            ("set MIN_ANGLE_LIMIT", _REG_MIN_ANGLE_LIMIT, min_tick, 2),
            ("set MAX_ANGLE_LIMIT", _REG_MAX_ANGLE_LIMIT, max_tick, 2),
            ("relock EEPROM", _REG_LOCK, 1, 1),
        ]
        for name, reg, val, width in steps:
            write = (
                self._packet.write1ByteTxRx if width == 1
                else self._packet.write2ByteTxRx
            )
            last_err = 0
            for _attempt in range(4):
                with self._lock:
                    try:
                        self._port.ser.reset_input_buffer()
                    except Exception:
                        pass
                    result, error = write(self._port, servo_id, reg, val)
                if result == 0:
                    if error != 0:
                        log.warning(f"{name} id={servo_id}: servo error 0x{error:02x}")
                    break
                last_err = result
                time.sleep(0.03)
            else:
                raise RuntimeError(f"{name} id={servo_id}: comm code={last_err}")
            # EEPROM commit pause — datasheet says ~10 ms, 30 ms is safe
            time.sleep(0.03)

    def _write_with_retry(self, op: str, writer) -> None:
        """Retry a write-register call up to 3 times on transient bus errors."""
        last_result = 0
        last_error = 0
        for _attempt in range(3):
            with self._lock:
                try:
                    self._port.ser.reset_input_buffer()
                except Exception:
                    pass
                result, error = writer()
            if result == 0:
                if error != 0:
                    log.warning(f"ST3215 {op} servo error byte: 0x{error:02x}")
                return
            last_result, last_error = result, error
        raise RuntimeError(f"ST3215 {op} comm error after retries: code={last_result}")

    def enable_torque(self, servo_id: int, on: bool) -> None:
        self._write_with_retry(
            "enable_torque",
            lambda: self._packet.write1ByteTxRx(
                self._port, servo_id, _REG_TORQUE_ENABLE, 1 if on else 0
            ),
        )

    def set_torque_limit(self, servo_id: int, limit: int) -> None:
        limit = max(0, min(1000, int(limit)))
        self._write_with_retry(
            "set_torque_limit",
            lambda: self._packet.write2ByteTxRx(
                self._port, servo_id, _REG_TORQUE_LIMIT, limit
            ),
        )

    def set_goal_speed(self, servo_id: int, speed: int) -> None:
        self._write_with_retry(
            "set_goal_speed",
            lambda: self._packet.write2ByteTxRx(
                self._port, servo_id, _REG_GOAL_SPEED, int(speed)
            ),
        )

    def set_goal_time(self, servo_id: int, time_units: int) -> None:
        """Sets GOAL_TIME register. When non-zero, the servo completes its move
        in approximately this many time units (typically ~1 unit ≈ 1 ms on
        STS3215, but firmware varies). With BOTH motors set to the same value,
        a coordinated move (pan, tilt, or combo) finishes simultaneously on
        both — preventing the mid-motion cross-coupling where one motor
        arrives early and the head transiently tilts while the other catches
        up. Set to 0 to disable time-mode and revert to GOAL_SPEED."""
        self._write_with_retry(
            "set_goal_time",
            lambda: self._packet.write2ByteTxRx(
                self._port, servo_id, _REG_GOAL_TIME, max(0, min(65535, int(time_units)))
            ),
        )

    def set_goal_acceleration(self, servo_id: int, accel: int) -> None:
        self._write_with_retry(
            "set_goal_acceleration",
            lambda: self._packet.write1ByteTxRx(
                self._port, servo_id, _REG_GOAL_ACCEL, max(0, min(254, int(accel)))
            ),
        )

    def read_position(self, servo_id: int) -> int:
        # Retry on transient bus corruption (see note on ping)
        last_err = 0
        for _attempt in range(3):
            with self._lock:
                try:
                    self._port.ser.reset_input_buffer()
                except Exception:
                    pass
                pos, result, error = self._packet.read2ByteTxRx(
                    self._port, servo_id, _REG_PRESENT_POSITION
                )
            if result == 0:
                if error != 0:
                    log.warning(f"read_position id={servo_id}: servo error 0x{error:02x}")
                return int(pos)
            last_err = result
        raise RuntimeError(f"read_position id={servo_id}: comm code={last_err}")

    def read_telemetry(self, servo_id: int) -> ServoTelemetry:
        with self._lock:
            try:
                pos, r1, _ = self._packet.read2ByteTxRx(self._port, servo_id, _REG_PRESENT_POSITION)
                spd, r2, _ = self._packet.read2ByteTxRx(self._port, servo_id, _REG_PRESENT_SPEED)
                load, r3, _ = self._packet.read2ByteTxRx(self._port, servo_id, _REG_PRESENT_LOAD)
                volt, r4, _ = self._packet.read1ByteTxRx(self._port, servo_id, _REG_PRESENT_VOLTAGE)
                temp, r5, _ = self._packet.read1ByteTxRx(self._port, servo_id, _REG_PRESENT_TEMPERATURE)
            except Exception as e:
                return ServoTelemetry(servo_id, 0, 0, 0, 0.0, 0.0, ok=False, error=str(e))
        ok = all(r == 0 for r in (r1, r2, r3, r4, r5))
        return ServoTelemetry(
            servo_id=servo_id,
            position_tick=int(pos),
            speed=int(spd),
            load=int(load),
            voltage_v=int(volt) / 10.0,
            temperature_c=float(temp),
            ok=ok,
        )

    def write_goal(self, servo_id: int, tick: int) -> None:
        tick = max(0, min(4095, int(tick)))
        self._write_with_retry(
            "write_goal",
            lambda: self._packet.write2ByteTxRx(
                self._port, servo_id, _REG_GOAL_POSITION, tick
            ),
        )

    def sync_write_goal(self, goals: dict[int, int]) -> None:
        """Sync-write goal positions to multiple servos in a single bus
        transaction — critical for coordinated L/R motion (avoids pitch wobble
        when commanding pure pan).
        """
        try:
            from scservo_sdk import GroupSyncWrite
        except ImportError:                                     # pragma: no cover
            # Fall back to individual writes (not ideal, but keeps things working)
            for sid, tick in goals.items():
                self.write_goal(sid, tick)
            return

        with self._lock:
            group = GroupSyncWrite(self._port, self._packet, _REG_GOAL_POSITION, 2)
            for sid, tick in goals.items():
                tick = max(0, min(4095, int(tick)))
                data = [tick & 0xFF, (tick >> 8) & 0xFF]
                group.addParam(sid, bytes(data))
            result = group.txPacket()
            group.clearParam()
            if result != 0:
                raise RuntimeError(f"sync_write_goal comm error: code={result}")


# ── Simulated bus ───────────────────────────────────────────────────────────

class _SimServoState:
    __slots__ = ("id", "pos", "goal", "speed", "torque", "temperature", "voltage", "load")

    def __init__(self, servo_id: int):
        self.id = servo_id
        self.pos = 2048.0          # float for smooth dynamics
        self.goal = 2048.0
        self.speed = 1000.0        # ticks/s toward goal
        self.torque = False
        self.temperature = 32.0
        self.voltage = 7.4
        self.load = 0


class SimulatedBus:
    """In-memory bus with first-order dynamics. Matches ST3215Bus interface
    exactly so HeadController and the calibration wizard work unmodified."""

    def __init__(self, servo_ids: list[int] | None = None):
        ids = servo_ids if servo_ids is not None else [1, 2]
        self._servos: dict[int, _SimServoState] = {sid: _SimServoState(sid) for sid in ids}
        self._lock = threading.Lock()
        self._last_tick_time = time.monotonic()
        self._open = False

    def open(self) -> None:
        self._open = True
        self._last_tick_time = time.monotonic()
        log.info(f"Simulated bus open with servos {list(self._servos.keys())}")

    def close(self) -> None:
        self._open = False

    def _step_dynamics(self) -> None:
        """Advance each servo's position toward its goal at `speed` ticks/s."""
        now = time.monotonic()
        dt = min(0.1, now - self._last_tick_time)              # clamp large gaps
        self._last_tick_time = now
        for s in self._servos.values():
            if not s.torque:
                continue
            max_step = s.speed * dt
            delta = s.goal - s.pos
            if abs(delta) <= max_step:
                s.pos = s.goal
                s.load = 0
            else:
                s.pos += max_step if delta > 0 else -max_step
                s.load = int(200 * (delta / abs(delta)))       # sign of motion

    def ping(self, servo_id: int) -> bool:
        return servo_id in self._servos

    def scan(self, id_range: range = range(1, 10)) -> list[int]:
        return [sid for sid in id_range if sid in self._servos]

    def set_id(self, old_id: int, new_id: int) -> None:
        with self._lock:
            if old_id not in self._servos:
                raise RuntimeError(f"SimulatedBus: no servo with id {old_id}")
            if new_id in self._servos and new_id != old_id:
                raise RuntimeError(f"SimulatedBus: id {new_id} already in use")
            state = self._servos.pop(old_id)
            state.id = new_id
            self._servos[new_id] = state

    def enable_torque(self, servo_id: int, on: bool) -> None:
        with self._lock:
            self._servos[servo_id].torque = on

    def set_torque_limit(self, servo_id: int, limit: int) -> None:
        pass                                                   # no-op in sim

    def set_goal_speed(self, servo_id: int, speed: int) -> None:
        with self._lock:
            self._servos[servo_id].speed = float(max(1, speed))

    def set_goal_time(self, servo_id: int, time_units: int) -> None:
        pass                                                   # no-op in sim

    def set_goal_acceleration(self, servo_id: int, accel: int) -> None:
        pass                                                   # first-order sim, no accel model

    def read_position(self, servo_id: int) -> int:
        with self._lock:
            self._step_dynamics()
            return int(round(self._servos[servo_id].pos))

    def read_telemetry(self, servo_id: int) -> ServoTelemetry:
        with self._lock:
            self._step_dynamics()
            s = self._servos[servo_id]
            return ServoTelemetry(
                servo_id=servo_id,
                position_tick=int(round(s.pos)),
                speed=int(s.speed if s.torque and s.pos != s.goal else 0),
                load=s.load,
                voltage_v=s.voltage,
                temperature_c=s.temperature,
                ok=True,
            )

    def write_goal(self, servo_id: int, tick: int) -> None:
        with self._lock:
            self._servos[servo_id].goal = float(max(0, min(4095, tick)))

    def sync_write_goal(self, goals: dict[int, int]) -> None:
        with self._lock:
            for sid, tick in goals.items():
                if sid in self._servos:
                    self._servos[sid].goal = float(max(0, min(4095, tick)))

    # Test/debug helpers (not part of ServoBus protocol)
    def _hand_set_position(self, servo_id: int, tick: int) -> None:
        """Simulate user hand-holding the head at a position (bypasses dynamics)."""
        with self._lock:
            self._servos[servo_id].pos = float(tick)
            self._servos[servo_id].goal = float(tick)
