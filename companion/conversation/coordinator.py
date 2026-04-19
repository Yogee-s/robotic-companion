"""Coordinator — cross-modal reactions that don't belong to any one subsystem.

Kept intentionally small for the v1 launch. Two responsibilities:

1. **Face lost mid-reply.** If the user walks out of frame while the
   robot is speaking, finish the current sentence but don't cancel.
   After 8 s of continued absence, publish a `FaceLost`-derived soft
   signal so the conversation manager can drop to IDLE_WATCHING without
   the user "hearing" the cutoff. Prevents the jarring "robot replies to
   empty air" / "robot cuts itself off" failure modes.

2. **Health degradation logging.** Every `HealthDegraded` event is
   logged with its severity. The manager / renderer already own the
   user-facing recovery speech and display fallbacks; the coordinator
   doesn't need to duplicate that logic.

Deferred behaviours (tone directive, two-person DOA salience, proactive
event integration, tool confirmation band) each add one subscriber here
when they ship.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from companion.core.events import FaceAppeared, FaceLost, HealthDegraded

log = logging.getLogger(__name__)

_ABSENCE_COOLDOWN_S = 8.0


class Coordinator:
    """Subscribes to cross-modal events and reacts."""

    def __init__(self, *, event_bus, conversation_manager) -> None:
        self._bus = event_bus
        self._cm = conversation_manager
        self._last_face_lost_at: Optional[float] = None
        self._last_face_seen_at: float = time.time()
        self._lock = threading.Lock()
        self._running = False
        self._watcher_thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._bus.subscribe(FaceAppeared, self._on_face_appeared)
        self._bus.subscribe(FaceLost, self._on_face_lost)
        self._bus.subscribe(HealthDegraded, self._on_health_degraded)
        # Small watcher thread to enforce the 8-second cutoff — we want
        # the decision to fire even if no further bus events arrive.
        self._watcher_thread = threading.Thread(
            target=self._absence_watcher, daemon=True, name="coordinator-watcher"
        )
        self._watcher_thread.start()
        log.info("Coordinator started")

    def stop(self, timeout_s: float = 2.0) -> None:
        self._running = False
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=timeout_s)
            self._watcher_thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── event subscribers ────────────────────────────────────────────────
    def _on_face_appeared(self, _ev: FaceAppeared) -> None:
        with self._lock:
            self._last_face_seen_at = time.time()
            self._last_face_lost_at = None

    def _on_face_lost(self, _ev: FaceLost) -> None:
        with self._lock:
            self._last_face_lost_at = time.time()

    def _on_health_degraded(self, ev: HealthDegraded) -> None:
        level = {
            "warn": logging.WARNING,
            "error": logging.ERROR,
            "fatal": logging.CRITICAL,
        }.get(ev.severity, logging.WARNING)
        log.log(level, "HealthDegraded: %s (%s) — %s", ev.subsystem, ev.severity, ev.detail)

    # ── absence watcher ─────────────────────────────────────────────────
    def _absence_watcher(self) -> None:
        """Once per second, if the user has been gone long enough during
        a SPEAKING turn, nudge the manager toward IDLE_WATCHING. We don't
        directly manipulate manager state — we interrupt, which the
        manager already handles cleanly by unwinding the turn.
        """
        from companion.conversation.states import ConversationState
        while self._running:
            time.sleep(1.0)
            with self._lock:
                lost_at = self._last_face_lost_at
            if lost_at is None:
                continue
            if (time.time() - lost_at) < _ABSENCE_COOLDOWN_S:
                continue
            try:
                state = self._cm.state
            except Exception:
                continue
            if state == ConversationState.SPEAKING:
                log.info("User absent for >%ds during SPEAKING — returning to idle.",
                         int(_ABSENCE_COOLDOWN_S))
                self._cm.handle_ui_action("sleep", {})
                # Avoid retriggering until next face-lost event
                with self._lock:
                    self._last_face_lost_at = None
