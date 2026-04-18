"""Calibration persistence — read/write motor section of config.yaml in place.

The wizard fills a CalibrationResult; `save_to_config_yaml` writes the
corresponding keys back under `motor:` while preserving the rest of the file
(including comments — we do a targeted line-level edit of only the fields we
own, avoiding a full re-dump that would blow away the file's structure).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    left_zero_tick: int
    right_zero_tick: int
    left_direction: int
    right_direction: int
    gear_ratio_measured: float
    invert_pan: bool
    invert_tilt: bool
    pan_limits_deg: list[float]
    tilt_limits_deg: list[float]
    backlash_deg: float = 1.0
    # Optional — calibration wizard may also overwrite these if user changed them
    left_servo_id: int = 1
    right_servo_id: int = 2


_FIELD_PATTERN = re.compile(r"^(\s*)([a-zA-Z_][a-zA-Z0-9_]*):\s.*$")


def _format_value(value: Any) -> str:
    """Format a Python value for in-place YAML replacement."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, float):
        # Avoid trailing "100.0" for integer-like floats? No — keep .0 for clarity.
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def save_to_config_yaml(result: CalibrationResult, yaml_path: str | Path) -> None:
    """In-place update: finds the `motor:` block and overwrites the known keys.

    Works on config.yaml even if it has comments — unknown keys (e.g.
    `max_speed_ticks_per_s`) are left untouched. New keys we add are appended
    inside the block.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    lines = path.read_text().splitlines()
    updates = {
        "left_servo_id": result.left_servo_id,
        "right_servo_id": result.right_servo_id,
        "left_zero_tick": result.left_zero_tick,
        "right_zero_tick": result.right_zero_tick,
        "left_direction": result.left_direction,
        "right_direction": result.right_direction,
        "gear_ratio_measured": result.gear_ratio_measured,
        "backlash_deg": result.backlash_deg,
        "invert_pan": result.invert_pan,
        "invert_tilt": result.invert_tilt,
        "pan_limits_deg": result.pan_limits_deg,
        "tilt_limits_deg": result.tilt_limits_deg,
    }

    # Find the start of the motor: block
    motor_start = None
    for i, line in enumerate(lines):
        if re.match(r"^motor:\s*$", line):
            motor_start = i
            break
    if motor_start is None:
        raise ValueError(
            "Could not find `motor:` block in config.yaml — add the motor "
            "section manually (see MotorConfig defaults) and re-run."
        )

    # Find the end — next top-level key (no indent) or EOF
    motor_end = len(lines)
    for j in range(motor_start + 1, len(lines)):
        ln = lines[j]
        if ln and not ln.startswith((" ", "\t", "#")):
            motor_end = j
            break

    # Replace keys already present inside the block
    block_indent = None
    handled = set()
    for k in range(motor_start + 1, motor_end):
        m = _FIELD_PATTERN.match(lines[k])
        if not m:
            continue
        indent, key = m.group(1), m.group(2)
        if key in updates:
            if block_indent is None:
                block_indent = indent
            # Preserve any trailing comment on the line
            comment_pos = lines[k].find("#")
            comment = ""
            if comment_pos >= 0:
                comment = "  " + lines[k][comment_pos:].strip()
            lines[k] = f"{indent}{key}: {_format_value(updates[key])}{comment}"
            handled.add(key)

    # Append any missing keys at the end of the block
    missing = [k for k in updates if k not in handled]
    if missing:
        if block_indent is None:
            block_indent = "  "
        insert_at = motor_end
        new_lines = [f"{block_indent}{k}: {_format_value(updates[k])}" for k in missing]
        lines = lines[:insert_at] + new_lines + lines[insert_at:]

    path.write_text("\n".join(lines) + "\n")
    log.info(f"Wrote calibration results to {path}")


def calibration_summary(result: CalibrationResult) -> str:
    """Human-readable summary for the wizard's final screen / logs."""
    d = asdict(result)
    pan_lo, pan_hi = d["pan_limits_deg"]
    tilt_lo, tilt_hi = d["tilt_limits_deg"]
    return (
        f"Calibration complete:\n"
        f"  Servo IDs   : L={d['left_servo_id']}  R={d['right_servo_id']}\n"
        f"  Zeros       : L={d['left_zero_tick']}  R={d['right_zero_tick']}\n"
        f"  Directions  : L={d['left_direction']:+d}  R={d['right_direction']:+d}\n"
        f"  Gear ratio  : {d['gear_ratio_measured']:.3f} (measured)\n"
        f"  Backlash    : {d['backlash_deg']:.2f}°\n"
        f"  Invert      : pan={d['invert_pan']}  tilt={d['invert_tilt']}\n"
        f"  Pan limits  : {pan_lo:+.1f}° .. {pan_hi:+.1f}°\n"
        f"  Tilt limits : {tilt_lo:+.1f}° .. {tilt_hi:+.1f}°\n"
    )
