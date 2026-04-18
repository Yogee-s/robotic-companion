"""Shared state types for the display subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Scene(Enum):
    FACE = "face"
    QUICK_GRID = "quick_grid"
    MORE_LIST = "more_list"


class ConversationalState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    SLEEP = "sleep"


@dataclass
class FaceState:
    """Everything the face renderer needs to draw one frame."""

    valence: float = 0.0       # -1 sad … +1 happy
    arousal: float = 0.0       # -1 calm … +1 excited
    talking: bool = False
    listening: bool = False
    thinking: bool = False
    sleep: bool = False
    privacy: bool = False      # cover the camera — face shows "blindfolded"
    blink_rate_hz: float = 0.3
    gaze_x: float = 0.0        # -1 left … +1 right (driven by DOA)
    current_viseme: str = "rest"
    scene: Scene = Scene.FACE


QUICK_GRID_ACTIONS = ("mute_mic", "stop_talking", "sleep", "more")

MORE_LIST_ACTIONS = (
    ("timer", "Timer"),
    ("remind_me", "Remind me"),
    ("volume", "Volume"),
    ("privacy", "Camera privacy"),
    ("memory", "What do you remember"),
    ("personality", "Personality"),
    ("status", "Status"),
    ("restart", "Restart"),
)
