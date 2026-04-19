"""Behavior package — coordinates motor + face display as one actuator.

Subsystem responsibility split:
* `engine.BehaviorEngine` — the 20 Hz tick. Polls sensors, subscribes to
  state/emotion/affect events, commands both `HeadController` (via
  `FaceTracker`) and the display `Renderer`.
* `tracking.GainSchedule` — per-conversation-state tracker tuning
  (kp, deadband) + a one-pole smoothing filter on the face-tracker target.

Everything else (idle animations, affect-tag ornaments, orient-to-sound,
multi-face salience) is deferred to a post-launch milestone and should
be added here as an additional helper module without structural change
to `engine.py`.
"""

from companion.behavior.engine import BehaviorEngine
from companion.behavior.tracking import GainSchedule

__all__ = ["BehaviorEngine", "GainSchedule"]
