"""Timer tool — in-process countdown backed by threading.Timer."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from companion.tools.registry import tool

log = logging.getLogger(__name__)

_notifier: Optional[Callable[[str], None]] = None


def set_notifier(callback: Callable[[str], None]) -> None:
    """Install the callback used when a timer fires (speaks via TTS)."""
    global _notifier
    _notifier = callback


@tool("timer", "Set a countdown timer for a given number of seconds.")
def start_timer(duration_seconds: int, label: str = "timer") -> str:
    if duration_seconds <= 0:
        return "Duration must be positive."

    def _fire() -> None:
        msg = f"Your {label} is up."
        if _notifier is not None:
            try:
                _notifier(msg)
            except Exception:
                log.exception("Timer notifier raised")

    threading.Timer(duration_seconds, _fire).start()
    minutes, seconds = divmod(duration_seconds, 60)
    if minutes and seconds:
        human = f"{minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}"
    elif minutes:
        human = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        human = f"{seconds} second{'s' if seconds != 1 else ''}"
    return f"Timer set for {human}."
