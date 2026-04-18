"""Stopwatch tool — start/stop/check."""

from __future__ import annotations

import time
from typing import Optional

from companion.tools.registry import tool

_started_at: Optional[float] = None


@tool("stopwatch_start", "Start the stopwatch.")
def start() -> str:
    global _started_at
    _started_at = time.time()
    return "Stopwatch started."


@tool("stopwatch_stop", "Stop the stopwatch and return the elapsed time.")
def stop() -> str:
    global _started_at
    if _started_at is None:
        return "Stopwatch wasn't running."
    elapsed = time.time() - _started_at
    _started_at = None
    mins, secs = divmod(int(elapsed), 60)
    return f"Stopped at {mins} minutes {secs} seconds."


@tool("stopwatch_check", "Check the current stopwatch reading without stopping.")
def check() -> str:
    if _started_at is None:
        return "Stopwatch isn't running."
    elapsed = time.time() - _started_at
    mins, secs = divmod(int(elapsed), 60)
    return f"{mins} minutes {secs} seconds so far."
