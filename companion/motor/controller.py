"""High-level head controller — pan/tilt API over a ServoBus.

One HeadController owns one ServoBus (live or simulated). Callers never touch
encoder ticks or per-motor signs; everything is in head-frame degrees.

Background poll thread reads telemetry at `poll_hz` and exposes the latest
pose + per-servo telemetry. A thermal watchdog disables torque if either
servo exceeds `max_temperature_c`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from companion.core.config import MotorConfig
from companion.motor.bus import ServoBus, ServoTelemetry, ST3215Bus, SimulatedBus
from companion.motor.kinematics import head_pose_to_ticks, ticks_to_head_pose

log = logging.getLogger(__name__)


@dataclass
class HeadState:
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    target_pan_deg: float = 0.0
    target_tilt_deg: float = 0.0
    left_goal_tick: int = 0
    right_goal_tick: int = 0
    left: Optional[ServoTelemetry] = None
    right: Optional[ServoTelemetry] = None
    torque_on: bool = False
    over_temp: bool = False
    stalled: bool = False                      # last commanded move couldn't be reached
    last_update_ts: float = field(default_factory=time.monotonic)


PoseCallback = Callable[[HeadState], None]


def make_bus(cfg: MotorConfig) -> ServoBus:
    """Factory — return a SimulatedBus or ST3215Bus based on config."""
    if cfg.sim_only:
        return SimulatedBus(servo_ids=[cfg.left_servo_id, cfg.right_servo_id])
    return ST3215Bus(port=cfg.port, baudrate=cfg.baudrate)


class HeadController:
    def __init__(self, cfg: MotorConfig, bus: Optional[ServoBus] = None):
        self.cfg = cfg
        self.bus: ServoBus = bus if bus is not None else make_bus(cfg)
        self.state = HeadState()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._pose_callbacks: list[PoseCallback] = []
        self._connected = False
        # Stall tracker: timestamp when the current goal started being unmet,
        # or None if the goal is currently reached (or hasn't been set).
        self._unreached_since: Optional[float] = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def connect(self) -> None:
        if self._connected:
            return
        self.bus.open()
        self._connected = True
        self._apply_motion_params()
        if self.cfg.home_on_startup:
            self.enable_torque(True)
            self.set_head_pose(0.0, 0.0)
        self._start_poll_thread()

    def disconnect(self) -> None:
        self._stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None
        if self._connected:
            try:
                self.enable_torque(False)
            except Exception as e:
                log.warning(f"disable torque on shutdown failed: {e}")
            self.bus.close()
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    # ── configuration push ─────────────────────────────────────────────────
    def _apply_motion_params(self) -> None:
        for sid in (self.cfg.left_servo_id, self.cfg.right_servo_id):
            try:
                self.bus.set_torque_limit(sid, self.cfg.torque_limit)
                self.bus.set_goal_speed(sid, self.cfg.max_speed_ticks_per_s)
                self.bus.set_goal_acceleration(sid, self.cfg.max_acceleration)
            except Exception as e:
                log.warning(f"apply_motion_params id={sid}: {e}")

    def reload_config(self, cfg: MotorConfig) -> None:
        """Swap in updated config (after calibration re-saves)."""
        self.cfg = cfg
        if self._connected:
            self._apply_motion_params()

    # ── motion ─────────────────────────────────────────────────────────────
    def enable_torque(self, on: bool) -> None:
        for sid in (self.cfg.left_servo_id, self.cfg.right_servo_id):
            self.bus.enable_torque(sid, on)
        with self._state_lock:
            self.state.torque_on = on

    def set_head_pose(self, pan_deg: float, tilt_deg: float) -> tuple[int, int]:
        """Command head to a target pan/tilt (degrees). Returns (left_tick, right_tick)
        actually written after limit clamping."""
        left_tick, right_tick = head_pose_to_ticks(pan_deg, tilt_deg, self.cfg)
        self._write_goal_ticks(left_tick, right_tick)
        with self._state_lock:
            self.state.target_pan_deg = max(
                self.cfg.pan_limits_deg[0], min(self.cfg.pan_limits_deg[1], pan_deg)
            )
            self.state.target_tilt_deg = max(
                self.cfg.tilt_limits_deg[0], min(self.cfg.tilt_limits_deg[1], tilt_deg)
            )
        return left_tick, right_tick

    def _write_goal_ticks(self, left_tick: int, right_tick: int) -> None:
        """Send goal ticks, update state, reset stall tracker."""
        goals = {
            self.cfg.left_servo_id: left_tick,
            self.cfg.right_servo_id: right_tick,
        }
        if self.cfg.sync_write:
            self.bus.sync_write_goal(goals)
        else:
            for sid, tick in goals.items():
                self.bus.write_goal(sid, tick)
        with self._state_lock:
            self.state.left_goal_tick = left_tick
            self.state.right_goal_tick = right_tick
            self.state.stalled = False
        self._unreached_since = None

    def home(self) -> None:
        self.set_head_pose(0.0, 0.0)

    def get_head_pose(self) -> tuple[float, float]:
        """Current head pose read live from encoders (blocks on bus I/O)."""
        left_tick = self.bus.read_position(self.cfg.left_servo_id)
        right_tick = self.bus.read_position(self.cfg.right_servo_id)
        return ticks_to_head_pose(left_tick, right_tick, self.cfg)

    # ── direct motor access (calibration wizard uses these) ────────────────
    def write_raw_ticks(self, left_tick: int, right_tick: int) -> None:
        """Send raw ticks — bypasses all kinematics. Only for calibration use."""
        self._write_goal_ticks(left_tick, right_tick)

    def read_raw_ticks(self) -> tuple[int, int]:
        return (
            self.bus.read_position(self.cfg.left_servo_id),
            self.bus.read_position(self.cfg.right_servo_id),
        )

    # ── telemetry polling ──────────────────────────────────────────────────
    def subscribe(self, cb: PoseCallback) -> None:
        self._pose_callbacks.append(cb)

    def unsubscribe(self, cb: PoseCallback) -> None:
        if cb in self._pose_callbacks:
            self._pose_callbacks.remove(cb)

    def _start_poll_thread(self) -> None:
        self._stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="HeadControllerPoll", daemon=True
        )
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        period = 1.0 / max(1.0, self.cfg.poll_hz)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._poll_once()
            except Exception as e:
                log.warning(f"head poll error: {e}")
            sleep_for = period - (time.monotonic() - t0)
            if sleep_for > 0:
                self._stop.wait(sleep_for)

    def _check_stall(self, left: ServoTelemetry, right: ServoTelemetry) -> bool:
        """If either motor has been stuck far from its goal longer than
        `stall_timeout_s`, command both to hold their current position so
        they stop pushing. Returns True once a stall has been handled."""
        if not self.cfg.stall_detect or self.state.stalled:
            return self.state.stalled
        if not (left.ok and right.ok) or not self.state.torque_on:
            self._unreached_since = None
            return False

        with self._state_lock:
            lg = self.state.left_goal_tick
            rg = self.state.right_goal_tick
        if lg == 0 and rg == 0:
            # No goal has ever been commanded — don't treat 'at 2048' as stalled
            self._unreached_since = None
            return False

        err = max(abs(left.position_tick - lg), abs(right.position_tick - rg))
        if err <= self.cfg.stall_position_error_ticks:
            self._unreached_since = None
            return False

        now = time.monotonic()
        if self._unreached_since is None:
            self._unreached_since = now
            return False
        if now - self._unreached_since < self.cfg.stall_timeout_s:
            return False

        # Stall confirmed — hold current position (don't disable torque; head
        # would fall under gravity). Firmware torque_limit already caps the
        # physical force, so this just stops sustained push current.
        log.warning(
            f"Stall detected — head not reaching goal after {self.cfg.stall_timeout_s}s "
            f"(L err={left.position_tick - lg:+d}, R err={right.position_tick - rg:+d}). "
            f"Holding current position."
        )
        try:
            goals = {
                self.cfg.left_servo_id: left.position_tick,
                self.cfg.right_servo_id: right.position_tick,
            }
            if self.cfg.sync_write:
                self.bus.sync_write_goal(goals)
            else:
                for sid, tick in goals.items():
                    self.bus.write_goal(sid, tick)
        except Exception as e:
            log.error(f"stall-hold write failed: {e}")
        with self._state_lock:
            self.state.left_goal_tick = left.position_tick
            self.state.right_goal_tick = right.position_tick
            self.state.stalled = True
        self._unreached_since = None
        return True

    def _poll_once(self) -> None:
        left = self.bus.read_telemetry(self.cfg.left_servo_id)
        right = self.bus.read_telemetry(self.cfg.right_servo_id)
        if left.ok and right.ok:
            pan, tilt = ticks_to_head_pose(left.position_tick, right.position_tick, self.cfg)
        else:
            pan, tilt = self.state.pan_deg, self.state.tilt_deg

        over_temp = (
            left.ok and left.temperature_c > self.cfg.max_temperature_c
        ) or (
            right.ok and right.temperature_c > self.cfg.max_temperature_c
        )
        if over_temp and not self.state.over_temp:
            hot = []
            if left.ok and left.temperature_c > self.cfg.max_temperature_c:
                hot.append(f"L={left.temperature_c:.1f}°C")
            if right.ok and right.temperature_c > self.cfg.max_temperature_c:
                hot.append(f"R={right.temperature_c:.1f}°C")
            log.error(f"Head motor thermal cutoff: {', '.join(hot)} — disabling torque")
            try:
                self.enable_torque(False)
            except Exception as e:
                log.error(f"failed to disable torque on overtemp: {e}")

        stalled = self._check_stall(left, right)

        with self._state_lock:
            self.state.pan_deg = pan
            self.state.tilt_deg = tilt
            self.state.left = left
            self.state.right = right
            self.state.over_temp = over_temp
            self.state.last_update_ts = time.monotonic()
            snapshot = HeadState(
                pan_deg=self.state.pan_deg,
                tilt_deg=self.state.tilt_deg,
                target_pan_deg=self.state.target_pan_deg,
                target_tilt_deg=self.state.target_tilt_deg,
                left_goal_tick=self.state.left_goal_tick,
                right_goal_tick=self.state.right_goal_tick,
                left=left,
                right=right,
                torque_on=self.state.torque_on,
                over_temp=over_temp,
                stalled=stalled,
                last_update_ts=self.state.last_update_ts,
            )

        for cb in list(self._pose_callbacks):
            try:
                cb(snapshot)
            except Exception as e:
                log.warning(f"pose callback error: {e}")
