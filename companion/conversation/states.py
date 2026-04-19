"""Conversation state machine — enumerated states + legal-transition table.

The ConversationManager moves through these states as a user turn flows
from microphone through STT/LLM/TTS back to the speaker. Centralizing
the states + transitions here keeps the manager's turn handler from
growing stringly-typed logic.

States
------
IDLE_WATCHING     — no user speech in flight; camera + VAD running.
LISTENING         — VAD speech-start fired; collecting audio.
CAPTURING_INTENT  — VAD speech-end fired; running STT to decide intent.
THINKING          — STT finalized; LLM generating reply.
SPEAKING          — First TTS chunk queued/playing.
RECOVERING        — After cancellation or error; short cooldown before re-arm.
"""

from __future__ import annotations

import logging
from typing import Iterable

log = logging.getLogger(__name__)


class ConversationState:
    IDLE_WATCHING = "idle_watching"
    LISTENING = "listening"
    CAPTURING_INTENT = "capturing_intent"
    THINKING = "thinking"
    SPEAKING = "speaking"
    RECOVERING = "recovering"

    # Legacy aliases, for back-compat with existing callers (`manager.state`
    # consumers in main.py and the GUI) that still expect the old names.
    IDLE = IDLE_WATCHING
    PROCESSING = THINKING

    ALL: tuple[str, ...] = (
        IDLE_WATCHING,
        LISTENING,
        CAPTURING_INTENT,
        THINKING,
        SPEAKING,
        RECOVERING,
    )


# Legal transitions. Any transition not listed here is rejected at
# runtime with a warning log (indicates a bug, not a crash condition).
_LEGAL: dict[str, frozenset[str]] = {
    ConversationState.IDLE_WATCHING: frozenset({
        ConversationState.LISTENING,
        ConversationState.RECOVERING,
    }),
    ConversationState.LISTENING: frozenset({
        ConversationState.CAPTURING_INTENT,
        ConversationState.IDLE_WATCHING,
        ConversationState.RECOVERING,
    }),
    ConversationState.CAPTURING_INTENT: frozenset({
        ConversationState.THINKING,
        ConversationState.IDLE_WATCHING,  # empty / mumble transcript
        ConversationState.RECOVERING,
    }),
    ConversationState.THINKING: frozenset({
        ConversationState.SPEAKING,
        ConversationState.IDLE_WATCHING,
        ConversationState.LISTENING,  # interrupt with new utterance
        ConversationState.RECOVERING,
    }),
    ConversationState.SPEAKING: frozenset({
        ConversationState.IDLE_WATCHING,
        ConversationState.LISTENING,  # barge-in
        ConversationState.RECOVERING,
    }),
    ConversationState.RECOVERING: frozenset({
        ConversationState.IDLE_WATCHING,
    }),
}


def is_legal_transition(current: str, target: str) -> bool:
    """True iff `current -> target` is allowed by the state machine."""
    if current == target:
        return True
    return target in _LEGAL.get(current, frozenset())


def assert_legal(current: str, target: str) -> None:
    """Log-warn on an illegal transition but do not raise.

    The manager still performs the transition (tolerance over correctness
    during live operation); a test asserts clean traces.
    """
    if not is_legal_transition(current, target):
        log.warning("Illegal state transition: %s -> %s", current, target)


def active_turn_states() -> Iterable[str]:
    """States where a turn is in flight (used by SceneWatcher to pause)."""
    return (
        ConversationState.LISTENING,
        ConversationState.CAPTURING_INTENT,
        ConversationState.THINKING,
        ConversationState.SPEAKING,
    )
