"""Tiny multi-subscriber dispatcher. Not a bus — just list-of-callbacks per signal.

    >>> sig = Signal()
    >>> sig.connect(print)
    >>> sig.emit("hello")   # prints hello

Used for loose coupling between vision pipeline, conversation manager, and
display subsystems without pulling in a full event bus framework.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)


class Signal:
    def __init__(self, name: str = "signal") -> None:
        self._name = name
        self._callbacks: list[Callable[..., Any]] = []
        self._lock = threading.Lock()

    def connect(self, callback: Callable[..., Any]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def disconnect(self, callback: Callable[..., Any]) -> None:
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def emit(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            subs = list(self._callbacks)
        for cb in subs:
            try:
                cb(*args, **kwargs)
            except Exception:
                log.exception(f"Signal {self._name} subscriber {cb!r} raised")
