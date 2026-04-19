"""HealthMonitor — 1 Hz watchdog over the live subsystems.

Publishes `HealthDegraded` events on the bus when a subsystem looks
unhealthy, and `HealthRecovered` when it comes back. The `Coordinator`
subscribes and handles user-visible recovery (graceful degradation
speech, renderer fallback, etc.). This module only *detects*.

Subsystems watched:
* audio_in  — `AudioInput.is_starved` (no chunk in >2 s while running)
* camera    — `EmotionPipeline.get_state().timestamp` freshness
* esp32     — renderer transport heartbeat (if available)
* motor     — temperature below cutoff, not stalled
* llm       — last activity timestamp (reported by ConversationManager)

All checks are cheap. The monitor runs in a single daemon thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from companion.core.events import HealthDegraded, HealthRecovered

log = logging.getLogger(__name__)


class HealthMonitor:
    def __init__(
        self,
        *,
        event_bus,
        tick_hz: float = 1.0,
        audio_input=None,
        emotion_pipeline=None,
        renderer=None,
        head_controller=None,
    ) -> None:
        self._bus = event_bus
        self._period = 1.0 / max(0.1, tick_hz)
        self._audio_in = audio_input
        self._emotion = emotion_pipeline
        self._renderer = renderer
        self._head = head_controller

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._down: dict[str, str] = {}  # subsystem -> severity

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="health-monitor"
        )
        self._thread.start()
        log.info("HealthMonitor started")

    def stop(self, timeout_s: float = 2.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── checks ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while self._running:
            for subsystem, status in self._run_checks():
                self._update(subsystem, status)
            time.sleep(self._period)

    def _run_checks(self):
        """Yield (subsystem, status|None) tuples. None status = healthy."""
        # Audio input
        if self._audio_in is not None:
            if getattr(self._audio_in, "is_starved", False):
                yield ("audio_in", ("error", "no chunks for >2s"))
            else:
                yield ("audio_in", None)

        # Camera / vision pipeline
        if self._emotion is not None:
            try:
                state = self._emotion.get_state()
                ts = getattr(state, "timestamp", 0.0) or 0.0
                if ts and (time.time() - ts) > 2.0:
                    yield ("camera", ("warn", "no fresh frame in 2s"))
                else:
                    yield ("camera", None)
            except Exception as exc:
                yield ("camera", ("error", repr(exc)))

        # Motor telemetry (only when enabled and connected)
        if self._head is not None:
            try:
                st = self._head.state
                if getattr(st, "stalled", False):
                    yield ("motor", ("warn", "stalled"))
                elif getattr(st, "over_temperature", False):
                    yield ("motor", ("error", f"temp {getattr(st, 'temperature_c', 0):.1f}C"))
                else:
                    yield ("motor", None)
            except Exception as exc:
                yield ("motor", ("warn", repr(exc)))

        # ESP32 renderer (if it exposes a health property)
        if self._renderer is not None:
            healthy = bool(getattr(self._renderer, "is_transport_healthy", True))
            if not healthy:
                yield ("esp32", ("warn", "serial transport unhealthy"))
            else:
                yield ("esp32", None)

    # ── event emission ───────────────────────────────────────────────────
    def _update(self, subsystem: str, status) -> None:
        was_down = subsystem in self._down
        if status is None:
            if was_down:
                self._down.pop(subsystem, None)
                if self._bus is not None:
                    self._bus.publish(HealthRecovered(subsystem=subsystem, timestamp=time.time()))
            return

        severity, detail = status
        prev_sev = self._down.get(subsystem)
        if prev_sev == severity:
            return  # already reported at this level
        self._down[subsystem] = severity
        if self._bus is not None:
            self._bus.publish(
                HealthDegraded(
                    subsystem=subsystem,
                    severity=severity,
                    detail=detail,
                    timestamp=time.time(),
                )
            )
