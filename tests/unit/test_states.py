"""State machine legal-transition tests."""

from companion.conversation.states import (
    ConversationState,
    is_legal_transition,
)


def test_idle_to_listening_is_legal():
    assert is_legal_transition(
        ConversationState.IDLE_WATCHING, ConversationState.LISTENING
    )


def test_idle_to_speaking_is_illegal():
    # Must go through CAPTURING_INTENT + THINKING first.
    assert not is_legal_transition(
        ConversationState.IDLE_WATCHING, ConversationState.SPEAKING
    )


def test_thinking_back_to_listening_is_legal_for_interrupt():
    assert is_legal_transition(
        ConversationState.THINKING, ConversationState.LISTENING
    )


def test_same_state_is_trivially_legal():
    for s in ConversationState.ALL:
        assert is_legal_transition(s, s)
