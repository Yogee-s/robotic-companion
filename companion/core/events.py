"""Typed events + the legacy `Signal` dispatcher.

The preferred integration path is pub/sub through `companion.core.event_bus.EventBus`
keyed on the event dataclasses declared below. `Signal` stays for back-compat
with existing subscribers (vision pipeline, scene watcher) that haven't
migrated yet.

Conventions
-----------
* Event dataclasses are past-tense verb phrases (`TurnStarted`, not `StartTurn`).
* `turn_id` is a short UUID created by the Turn object.
* Payloads are immutable where possible (`frozen=True`) to prevent
  handlers from mutating shared state.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# ── Audio ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpeechStart:
    timestamp: float


@dataclass(frozen=True)
class SpeechEnd:
    timestamp: float
    duration_s: float


@dataclass(frozen=True)
class DOAUpdate:
    angle_deg: float           # 0 = body forward, positive = clockwise
    voice_active: bool
    timestamp: float


@dataclass(frozen=True)
class VisemeStream:
    """Audio envelope viseme events tied to a single TTS chunk."""
    turn_id: str
    events: list                # list[lip_sync.VisemeEvent]
    sample_rate: int
    timestamp: float


# ── Vision ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FaceAppeared:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    timestamp: float


@dataclass(frozen=True)
class FaceLost:
    timestamp: float


@dataclass(frozen=True)
class EmotionChanged:
    label: str
    valence: float             # -1 .. +1
    arousal: float             # -1 .. +1
    confidence: float          # 0 .. 1
    timestamp: float


# ── Conversation / Turn ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TurnStarted:
    turn_id: str
    transcript_preview: str = ""


@dataclass(frozen=True)
class TurnFirstToken:
    turn_id: str
    token: str


@dataclass(frozen=True)
class TurnFirstAudio:
    turn_id: str


@dataclass(frozen=True)
class TurnCancelled:
    turn_id: str
    reason: str


@dataclass(frozen=True)
class TurnCompleted:
    turn_id: str
    trace: Any                 # TurnTrace; Any here to avoid cross-module import


@dataclass(frozen=True)
class StateChanged:
    old: str
    new: str
    timestamp: float


@dataclass(frozen=True)
class AffectTag:
    """Affect cue parsed from `[affect: ...]` at the end of an LLM reply.

    Consumed by the BehaviorEngine to fire a one-shot ornament / expression.
    """
    tag: str                   # "happy" | "curious" | "confused" | ...


# ── Health / ops ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HealthDegraded:
    subsystem: str             # "audio_in" | "camera" | "esp32" | "llm" | "motor" | "tts"
    severity: str              # "warn" | "error" | "fatal"
    detail: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class HealthRecovered:
    subsystem: str
    timestamp: float = 0.0


# ── Legacy dispatcher (kept for current subscribers) ────────────────────────

class Signal:
    """Tiny multi-subscriber dispatcher. Prefer `EventBus` for new code."""

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
                log.exception("Signal %s subscriber %r raised", self._name, cb)
