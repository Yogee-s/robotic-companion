"""Differential bevel gear kinematics — pure functions, no I/O.

Mechanism: two ST3215 motors on the sides drive small pinion bevel gears
(20 teeth each); both pinions mesh with one large crown bevel gear (40 teeth)
carrying the head. The differential mixing gives two output DOFs (pan, tilt)
from two motor inputs.

Convention used everywhere in this module:
    gear_ratio = crown_teeth / pinion_teeth  (reduction, typically > 1)

Standard two-input bevel differential:
    Forward (motor → head):
        pitch = (θ_L + θ_R) / (2 · gear_ratio)
        pan   = (θ_L − θ_R) / (2 · gear_ratio)
    Inverse (head → motor):
        θ_L = gear_ratio · (pitch + pan)
        θ_R = gear_ratio · (pitch − pan)

Sign conventions for motor direction, pan axis, and tilt axis are resolved
via calibration (see companion.motor.calibration) — they live on MotorConfig.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from companion.core.config import MotorConfig


TICKS_PER_REV = 4096                                   # ST3215 absolute encoder
_TICKS_PER_RAD = TICKS_PER_REV / (2.0 * math.pi)
_RADS_PER_TICK = (2.0 * math.pi) / TICKS_PER_REV


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def head_pose_to_ticks(pan_deg: float, tilt_deg: float, cfg: "MotorConfig") -> tuple[int, int]:
    """Inverse kinematics — head pose (deg) → motor encoder ticks.

    Clamps inputs to the soft limits defined in `cfg` before computing. Returns
    integer ticks ready for sync-write to the servos.
    """
    pan_deg = _clamp(pan_deg, cfg.pan_limits_deg[0], cfg.pan_limits_deg[1])
    tilt_deg = _clamp(tilt_deg, cfg.tilt_limits_deg[0], cfg.tilt_limits_deg[1])

    if cfg.invert_pan:
        pan_deg = -pan_deg
    if cfg.invert_tilt:
        tilt_deg = -tilt_deg

    pan_rad = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_deg)
    g = cfg.gear_ratio_measured

    theta_l = g * (tilt_rad + pan_rad)
    theta_r = g * (tilt_rad - pan_rad)

    theta_l *= cfg.left_direction
    theta_r *= cfg.right_direction

    left_ticks = int(round(theta_l * _TICKS_PER_RAD + cfg.left_zero_tick))
    right_ticks = int(round(theta_r * _TICKS_PER_RAD + cfg.right_zero_tick))
    return left_ticks, right_ticks


def ticks_to_head_pose(left_ticks: int, right_ticks: int, cfg: "MotorConfig") -> tuple[float, float]:
    """Forward kinematics — motor encoder ticks → head pose (deg)."""
    theta_l = (left_ticks - cfg.left_zero_tick) * _RADS_PER_TICK
    theta_r = (right_ticks - cfg.right_zero_tick) * _RADS_PER_TICK

    theta_l *= cfg.left_direction
    theta_r *= cfg.right_direction

    g = cfg.gear_ratio_measured
    tilt_rad = (theta_l + theta_r) / (2.0 * g)
    pan_rad = (theta_l - theta_r) / (2.0 * g)

    pan_deg = math.degrees(pan_rad)
    tilt_deg = math.degrees(tilt_rad)

    if cfg.invert_pan:
        pan_deg = -pan_deg
    if cfg.invert_tilt:
        tilt_deg = -tilt_deg
    return pan_deg, tilt_deg


def ticks_to_degrees(ticks: int) -> float:
    """Raw tick delta → degrees, around one motor shaft (no gear reduction)."""
    return ticks * _RADS_PER_TICK * 180.0 / math.pi


def degrees_to_ticks(deg: float) -> int:
    """Raw degree delta → ticks, around one motor shaft (no gear reduction)."""
    return int(round(math.radians(deg) * _TICKS_PER_RAD))
