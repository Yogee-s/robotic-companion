"""EmotionState + conversation state → FaceState mapping.

Takes the current valence/arousal and the current conversation state and
produces the renderer's input. Keeps mapping policy in one place so the
same derivation is used whether we're drawing on the Jetson (pygame) or
the ESP32 (via serial commands).
"""

from __future__ import annotations

from companion.display.state import ConversationalState, FaceState, Scene
from companion.vision.pipeline import EmotionState


def emotion_to_face(
    emotion: EmotionState,
    conv: ConversationalState,
    doa_angle_deg: float | None = None,
    privacy: bool = False,
) -> FaceState:
    fs = FaceState(
        valence=float(emotion.valence),
        arousal=float(emotion.arousal),
        talking=conv == ConversationalState.SPEAKING,
        listening=conv == ConversationalState.LISTENING,
        thinking=conv == ConversationalState.THINKING,
        sleep=conv == ConversationalState.SLEEP,
        privacy=privacy,
        blink_rate_hz=0.3 + max(0.0, emotion.arousal) * 0.2,
        scene=Scene.FACE,
    )
    if doa_angle_deg is not None:
        # ReSpeaker DOA is degrees (0 = front, -180..+180). Map to -1..+1.
        clamped = max(-90.0, min(90.0, doa_angle_deg))
        fs.gaze_x = clamped / 90.0
    return fs
