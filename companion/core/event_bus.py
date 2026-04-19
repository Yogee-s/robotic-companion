"""EventBus — the pub/sub spine that wires subsystems together.

Design goals
------------
* **Simple and synchronous by default.** `publish(topic, payload)` calls
  subscribers in-line; payload is a typed event dataclass from
  `companion.core.events`. Most handlers are cheap (state updates, log
  writes); the few that aren't use `publish_async` which dispatches to a
  small bounded thread pool.
* **Weak isolation.** A raising subscriber is logged but never breaks the
  publisher or the other subscribers.
* **Topic is the event class itself.** Publishers call
  `bus.publish(TurnStarted(turn_id=...))`; subscribers call
  `bus.subscribe(TurnStarted, handler)`. String topics are error-prone;
  classes give us dataclass type-checking at the call site.
* **Lifecycle.** `start()` spins up the async worker pool; `stop()` drains
  and joins. `shutdown()` is idempotent. Nothing runs until `start()`.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import defaultdict
from typing import Any, Callable, Type

log = logging.getLogger(__name__)

# Subscribers take the event dataclass and return nothing. We accept any
# callable to stay flexible for adapter lambdas.
Handler = Callable[[Any], None]


class EventBus:
    """In-process pub/sub keyed by event-class type.

    Subscribers may be registered before or after `start()`. Publishes
    before `start()` are allowed (sync subscribers fire; async publishes
    enqueue and are drained when the workers come up).
    """

    def __init__(self, async_workers: int = 2, async_queue_size: int = 256) -> None:
        self._subs: dict[Type[Any], list[Handler]] = defaultdict(list)
        self._subs_lock = threading.RLock()
        self._async_workers = max(1, int(async_workers))
        self._async_queue: queue.Queue = queue.Queue(maxsize=async_queue_size)
        self._workers: list[threading.Thread] = []
        self._running = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self._async_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"eventbus-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def stop(self, timeout_s: float = 2.0) -> None:
        if not self._running:
            return
        self._running = False
        # Poison pills so workers can exit their blocking get()
        for _ in self._workers:
            try:
                self._async_queue.put_nowait(None)
            except queue.Full:
                pass
        for t in self._workers:
            t.join(timeout=timeout_s / max(1, len(self._workers)))
        self._workers.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    # ── subscribe / publish ──────────────────────────────────────────────
    def subscribe(self, event_type: Type[Any], handler: Handler) -> None:
        """Register a handler for events of exactly this class."""
        with self._subs_lock:
            self._subs[event_type].append(handler)

    def unsubscribe(self, event_type: Type[Any], handler: Handler) -> None:
        with self._subs_lock:
            try:
                self._subs[event_type].remove(handler)
            except ValueError:
                pass

    def publish(self, event: Any) -> None:
        """Synchronously fan out to subscribers of the event's class."""
        event_type = type(event)
        with self._subs_lock:
            handlers = list(self._subs.get(event_type, ()))
        for h in handlers:
            try:
                h(event)
            except Exception:
                log.exception("Subscriber %r raised for %s", h, event_type.__name__)

    def publish_async(self, event: Any) -> None:
        """Hand the event off to the worker pool. Never blocks the publisher.

        On queue full we drop oldest + log. Losing a telemetry event is
        preferable to stalling a real-time turn thread.
        """
        try:
            self._async_queue.put_nowait(event)
        except queue.Full:
            try:
                _ = self._async_queue.get_nowait()
                self._async_queue.put_nowait(event)
                log.warning("EventBus async queue saturated; dropped oldest")
            except queue.Empty:
                pass

    # ── internals ────────────────────────────────────────────────────────
    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._async_queue.get(timeout=0.25)
            except queue.Empty:
                if not self._running:
                    return
                continue
            if item is None:  # poison pill
                return
            self.publish(item)
