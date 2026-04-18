"""Reminder tool — persistent file-backed reminders.

Stores reminders under `data/reminders.json` so they survive restarts.
On startup, `load_pending()` schedules any still-future reminders.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, Optional

from companion.tools.registry import tool

log = logging.getLogger(__name__)

_STORE = "data/reminders.json"
_notifier: Optional[Callable[[str], None]] = None


def set_notifier(callback: Callable[[str], None]) -> None:
    global _notifier
    _notifier = callback


def _load() -> list[dict]:
    if not os.path.exists(_STORE):
        return []
    try:
        with open(_STORE) as fh:
            return json.load(fh)
    except Exception:
        return []


def _save(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(_STORE) or ".", exist_ok=True)
    with open(_STORE, "w") as fh:
        json.dump(entries, fh, indent=2)


def _schedule(entry: dict) -> None:
    delay = entry["fire_at"] - time.time()
    if delay <= 0:
        _fire(entry)
        return
    threading.Timer(delay, _fire, args=(entry,)).start()


def _fire(entry: dict) -> None:
    msg = f"Reminder: {entry['text']}"
    if _notifier is not None:
        try:
            _notifier(msg)
        except Exception:
            log.exception("Reminder notifier raised")
    entries = [e for e in _load() if e.get("id") != entry.get("id")]
    _save(entries)


def load_pending() -> None:
    now = time.time()
    for e in _load():
        if e.get("fire_at", 0) > now:
            _schedule(e)


@tool("remind_me", "Set a reminder with a text message and delay in seconds.")
def remind_me(text: str, delay_seconds: int) -> str:
    fire_at = time.time() + int(delay_seconds)
    entry = {"id": int(time.time() * 1000), "text": text, "fire_at": fire_at}
    entries = _load() + [entry]
    _save(entries)
    _schedule(entry)
    return f"I'll remind you: {text}."
