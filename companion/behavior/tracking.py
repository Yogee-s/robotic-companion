"""Per-state gain schedule for FaceTracker.

The robot tracks more tightly when it's listening (user is engaged, we
want to really attend to them) and holds still while thinking (moving
mid-LLM-generation makes it look restless). Gain and deadband vary by
conversation state.

The values are deliberately conservative — the point is smoother,
"calmer" motion, not snappy recovery. If you want sharper tracking,
bump `LISTENING.kp` and leave the rest alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from companion.conversation.states import ConversationState


@dataclass(frozen=True)
class TrackingProfile:
    kp: float
    deadband_deg: float


# Tuned for "the robot is present but not twitchy" — change one value at
# a time and observe; changing several together makes it hard to tell
# which parameter caused the behavior shift.
_PROFILES: dict[str, TrackingProfile] = {
    ConversationState.IDLE_WATCHING: TrackingProfile(kp=0.30, deadband_deg=4.0),
    ConversationState.LISTENING:     TrackingProfile(kp=0.45, deadband_deg=2.5),
    ConversationState.CAPTURING_INTENT: TrackingProfile(kp=0.30, deadband_deg=3.5),
    # Head holds still while the LLM is composing a reply. kp=0 freezes
    # pan/tilt at whatever they were last commanded to.
    ConversationState.THINKING:      TrackingProfile(kp=0.0,  deadband_deg=0.0),
    ConversationState.SPEAKING:      TrackingProfile(kp=0.25, deadband_deg=3.0),
    ConversationState.RECOVERING:    TrackingProfile(kp=0.20, deadband_deg=4.0),
}

_DEFAULT = TrackingProfile(kp=0.30, deadband_deg=4.0)


class GainSchedule:
    """Applies the per-state tracking profile to a `FaceTracker` instance."""

    def __init__(self, tracker) -> None:
        self._tracker = tracker
        self._current_state: str = ConversationState.IDLE_WATCHING

    def apply(self, state: str) -> None:
        """Apply the profile for `state` to the underlying tracker."""
        if state == self._current_state:
            return
        self._current_state = state
        profile = _PROFILES.get(state, _DEFAULT)
        self._tracker.kp = profile.kp
        self._tracker.deadband_deg = profile.deadband_deg

    def profile_for(self, state: str) -> TrackingProfile:
        return _PROFILES.get(state, _DEFAULT)
