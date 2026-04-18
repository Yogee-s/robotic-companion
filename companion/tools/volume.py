"""Volume tool — uses `amixer` for ALSA master control on Jetson."""

from __future__ import annotations

import logging
import subprocess

from companion.tools.registry import tool

log = logging.getLogger(__name__)


def _set_alsa_master(percent: int) -> bool:
    percent = max(0, min(100, int(percent)))
    try:
        subprocess.run(
            ["amixer", "-q", "sset", "Master", f"{percent}%"],
            check=True,
            timeout=3.0,
        )
        return True
    except Exception as exc:
        log.debug(f"amixer set failed: {exc!r}")
        return False


@tool("set_volume", "Set the speaker volume as a percentage 0-100.")
def set_volume(percent: int) -> str:
    if _set_alsa_master(percent):
        return f"Volume set to {percent}%."
    return "Sorry, I couldn't change the volume."


@tool("volume_up", "Increase the speaker volume by 10 percent.")
def volume_up() -> str:
    try:
        result = subprocess.run(
            ["amixer", "sget", "Master"], capture_output=True, text=True, timeout=3.0
        )
        import re
        m = re.search(r"\[(\d+)%\]", result.stdout)
        current = int(m.group(1)) if m else 50
    except Exception:
        current = 50
    return set_volume(min(100, current + 10))


@tool("volume_down", "Decrease the speaker volume by 10 percent.")
def volume_down() -> str:
    try:
        result = subprocess.run(
            ["amixer", "sget", "Master"], capture_output=True, text=True, timeout=3.0
        )
        import re
        m = re.search(r"\[(\d+)%\]", result.stdout)
        current = int(m.group(1)) if m else 50
    except Exception:
        current = 50
    return set_volume(max(0, current - 10))
