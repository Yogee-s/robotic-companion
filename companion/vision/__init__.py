"""Vision subsystem — camera, face detection, emotion, VLM scene understanding."""

from companion.vision.camera import CSICamera
from companion.vision.emotion_classifier import (
    EMOTION_LABELS,
    EMOTION_VA,
    EmotionClassifier,
)
from companion.vision.face_detector import FaceDetector
from companion.vision.face_tracker import (
    FaceTracker,
    TrackerSnapshot,
    render_annotated_frame,
)
from companion.vision.pipeline import EmotionPipeline, EmotionState

__all__ = [
    "CSICamera",
    "FaceDetector",
    "EmotionClassifier",
    "EMOTION_LABELS",
    "EMOTION_VA",
    "EmotionPipeline",
    "EmotionState",
    "FaceTracker",
    "TrackerSnapshot",
    "render_annotated_frame",
]
