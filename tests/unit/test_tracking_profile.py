"""Tracking gain schedule — per-state kp + deadband."""

from companion.behavior.tracking import GainSchedule
from companion.conversation.states import ConversationState


class _FakeTracker:
    def __init__(self) -> None:
        self.kp = 0.0
        self.deadband_deg = 0.0


def test_thinking_state_freezes_head():
    fake = _FakeTracker()
    sched = GainSchedule(fake)
    # Force the state away from the default first, then into THINKING
    sched.apply(ConversationState.LISTENING)
    sched.apply(ConversationState.THINKING)
    assert fake.kp == 0.0


def test_listening_is_more_attentive_than_idle():
    sched = GainSchedule(_FakeTracker())
    idle = sched.profile_for(ConversationState.IDLE_WATCHING)
    lis = sched.profile_for(ConversationState.LISTENING)
    assert lis.kp > idle.kp
    assert lis.deadband_deg <= idle.deadband_deg


def test_same_state_is_idempotent():
    fake = _FakeTracker()
    sched = GainSchedule(fake)
    sched.apply(ConversationState.LISTENING)
    kp = fake.kp
    fake.kp = 999.0  # external change
    sched.apply(ConversationState.LISTENING)  # no re-apply
    assert fake.kp == 999.0  # left alone because state didn't change
