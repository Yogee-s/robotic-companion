"""Kinematics round-trip and invariants for the differential bevel head.

Run:
    python -m pytest tests/test_motor_kinematics.py -q
or (without pytest):
    python tests/test_motor_kinematics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow running the file directly (no pytest) as `python tests/test_motor_kinematics.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from companion.core.config import MotorConfig
from companion.motor.kinematics import (
    head_pose_to_ticks, ticks_to_head_pose,
    degrees_to_ticks, ticks_to_degrees,
    TICKS_PER_REV,
)


def _make_cfg(**overrides) -> MotorConfig:
    cfg = MotorConfig()
    cfg.pan_limits_deg = [-90.0, 90.0]
    cfg.tilt_limits_deg = [-30.0, 30.0]
    cfg.gear_ratio_measured = 2.0
    cfg.left_zero_tick = 2048
    cfg.right_zero_tick = 2048
    cfg.left_direction = 1
    cfg.right_direction = -1
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_round_trip_identity():
    cfg = _make_cfg()
    for pan in range(-80, 81, 10):
        for tilt in range(-25, 26, 5):
            lt, rt = head_pose_to_ticks(pan, tilt, cfg)
            p2, t2 = ticks_to_head_pose(lt, rt, cfg)
            assert abs(p2 - pan) < 0.1, f"pan round-trip: {pan} → {p2}"
            assert abs(t2 - tilt) < 0.1, f"tilt round-trip: {tilt} → {t2}"


def test_zero_pose_at_zero_ticks():
    cfg = _make_cfg()
    lt, rt = head_pose_to_ticks(0.0, 0.0, cfg)
    assert lt == cfg.left_zero_tick
    assert rt == cfg.right_zero_tick
    pan, tilt = ticks_to_head_pose(cfg.left_zero_tick, cfg.right_zero_tick, cfg)
    assert abs(pan) < 1e-6
    assert abs(tilt) < 1e-6


def test_soft_limits_clamp():
    cfg = _make_cfg()
    # Commanding beyond pan_max clamps to pan_max, not free
    lt1, rt1 = head_pose_to_ticks(200, 0, cfg)
    lt2, rt2 = head_pose_to_ticks(90, 0, cfg)
    assert lt1 == lt2 and rt1 == rt2
    lt3, rt3 = head_pose_to_ticks(-200, 0, cfg)
    lt4, rt4 = head_pose_to_ticks(-90, 0, cfg)
    assert lt3 == lt4 and rt3 == rt4


def test_pure_pan_gives_opposite_motor_deltas():
    """For pure pan (tilt=0), motor deltas should be equal in magnitude,
    opposite in sign (once per-motor direction sign is removed)."""
    cfg = _make_cfg()
    lt, rt = head_pose_to_ticks(30.0, 0.0, cfg)
    dl = (lt - cfg.left_zero_tick) * cfg.left_direction
    dr = (rt - cfg.right_zero_tick) * cfg.right_direction
    assert abs(dl + dr) < 2, f"pure pan should have dl ≈ -dr; got dl={dl} dr={dr}"


def test_pure_tilt_gives_equal_motor_deltas():
    """For pure tilt, motor deltas should be equal in sign and magnitude
    (after per-motor sign correction)."""
    cfg = _make_cfg()
    lt, rt = head_pose_to_ticks(0.0, 20.0, cfg)
    dl = (lt - cfg.left_zero_tick) * cfg.left_direction
    dr = (rt - cfg.right_zero_tick) * cfg.right_direction
    assert abs(dl - dr) < 2, f"pure tilt should have dl ≈ dr; got dl={dl} dr={dr}"


def test_gear_ratio_scales_motor_travel():
    """Doubling the gear ratio should double the motor travel for the same head pose."""
    cfg1 = _make_cfg(gear_ratio_measured=2.0)
    cfg2 = _make_cfg(gear_ratio_measured=4.0)
    lt1, rt1 = head_pose_to_ticks(10.0, 0.0, cfg1)
    lt2, rt2 = head_pose_to_ticks(10.0, 0.0, cfg2)
    dl1 = (lt1 - cfg1.left_zero_tick) * cfg1.left_direction
    dl2 = (lt2 - cfg2.left_zero_tick) * cfg2.left_direction
    assert abs(dl2 / dl1 - 2.0) < 0.05, f"expected 2× motor travel, got {dl2/dl1:.3f}"


def test_invert_pan_flips_sign():
    cfg_a = _make_cfg(invert_pan=False)
    cfg_b = _make_cfg(invert_pan=True)
    lt_a, rt_a = head_pose_to_ticks(20.0, 0.0, cfg_a)
    lt_b, rt_b = head_pose_to_ticks(20.0, 0.0, cfg_b)
    # With invert_pan, the motors should swap relative to the un-inverted case
    assert (lt_a - cfg_a.left_zero_tick) == -(lt_b - cfg_b.left_zero_tick)
    assert (rt_a - cfg_a.right_zero_tick) == -(rt_b - cfg_b.right_zero_tick)


def test_nonzero_motor_zeros():
    """Calibration shifts motor zeros off 2048 — round-trip must still hold."""
    cfg = _make_cfg(left_zero_tick=2300, right_zero_tick=1800)
    for pan in (-45, -10, 0, 25, 60):
        for tilt in (-15, 0, 10):
            lt, rt = head_pose_to_ticks(pan, tilt, cfg)
            p2, t2 = ticks_to_head_pose(lt, rt, cfg)
            assert abs(p2 - pan) < 0.1
            assert abs(t2 - tilt) < 0.1


def test_degrees_to_ticks_roundtrip():
    for deg in (0.0, 1.0, 45.0, 90.0, 179.0):
        tk = degrees_to_ticks(deg)
        back = ticks_to_degrees(tk)
        assert abs(back - deg) < 0.1


def test_one_motor_rev_equals_one_head_pitch_rev_over_gear_ratio():
    """Invariant: command pure pitch of (360°/gear_ratio); motor should rotate 360°."""
    g = 2.0
    cfg = _make_cfg(gear_ratio_measured=g, tilt_limits_deg=[-360.0, 360.0])
    lt, rt = head_pose_to_ticks(180.0 / g, 0.0, cfg)  # half-rev of head
    dl_deg = ticks_to_degrees((lt - cfg.left_zero_tick) * cfg.left_direction)
    assert abs(abs(dl_deg) - 180.0) < 1.0, f"expected ~180° motor, got {dl_deg}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"✗ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
