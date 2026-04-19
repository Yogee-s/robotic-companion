"""Turn — owns the lifecycle of one user↔robot exchange.

A Turn begins when VAD declares speech-end with the engagement gates
passing. It ends when TTS finishes or is cancelled. It carries:

* a cancellation token that tears down STT / LLM / TTS in flight,
* a TurnTrace for latency telemetry,
* the partial + final transcript,
* the emotion snapshot captured at start,
* handles to the generation and streaming futures.

ConversationManager holds at most one live Turn (`self._current_turn`).
A new user utterance arriving mid-turn cancels the old one via
`turn.cancel("new_utterance")` and spawns a fresh Turn.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from companion.core.telemetry import TurnTrace

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """State of a single conversation turn."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    cancel_event: threading.Event = field(default_factory=threading.Event)
    trace: TurnTrace = field(default=None)  # type: ignore[assignment]

    audio: Optional[Any] = None             # np.ndarray when finalised
    partial_transcript: str = ""
    final_transcript: str = ""

    emotion_label: Optional[str] = None
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    emotion_confidence: float = 0.0

    route: str = ""                         # chat | vqa | tool
    tool_name: Optional[str] = None
    reply_text: str = ""

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = TurnTrace(turn_id=self.turn_id)

    # ── cancellation ────────────────────────────────────────────────────
    def cancel(self, reason: str = "cancelled") -> None:
        """Mark this turn as cancelled and record the reason."""
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.trace.cancelled = True
        self.trace.interrupt_reason = reason
        log.info("Turn %s cancelled: %s", self.turn_id, reason)

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    # ── phase stamping ──────────────────────────────────────────────────
    def mark(self, phase: str) -> None:
        self.trace.mark(phase)

    # ── diagnostics ─────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"Turn(id={self.turn_id}, route={self.route!r}, "
            f"cancelled={self.is_cancelled})"
        )
