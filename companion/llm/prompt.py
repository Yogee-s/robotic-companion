"""System prompt + per-turn hint injection.

Gates:
- Emotion hint is injected only when confident (≥0.5) AND meaningfully
  changed (label change OR |Δvalence| ≥ 0.3) since the last hint — avoids
  spamming the small context window with redundant signal.
- Scene hint is injected lazily, refreshed at most every 20 seconds.
- Retrieved memories are listed as a short `[remembered: …]` line.

These helpers are pure — no I/O. The conversation manager calls them
before each LLM turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from companion.vision.pipeline import EmotionState


@dataclass
class EmotionHint:
    label: str
    confidence: float
    valence: float
    arousal: float

    @classmethod
    def from_state(cls, s: EmotionState) -> "EmotionHint":
        return cls(s.label, s.confidence, s.valence, s.arousal)


def build_system_prompt(
    base: str,
    verbosity: str = "normal",
    singlish: bool = False,
    speaker_name: Optional[str] = None,
) -> str:
    parts = [base.strip()]
    if verbosity == "brief":
        parts.append("Be very brief — one sentence at most.")
    elif verbosity == "detailed":
        parts.append("You may give slightly longer answers (up to 4 sentences).")
    if singlish:
        parts.append("Reply in casual Singlish — use 'lah', 'lor', 'leh' naturally.")
    if speaker_name:
        parts.append(f"You are currently talking to {speaker_name}.")
    return " ".join(parts)


def format_emotion_hint(
    current: Optional[EmotionHint],
    last: Optional[EmotionHint],
    *,
    confidence_floor: float = 0.5,
    valence_delta: float = 0.3,
) -> Optional[str]:
    """Return the hint string to prepend, or None to skip injection."""
    if current is None or current.confidence < confidence_floor:
        return None
    if last is not None:
        label_unchanged = current.label == last.label
        valence_stable = abs(current.valence - last.valence) < valence_delta
        if label_unchanged and valence_stable:
            return None
    return (
        f"[user_emotion: {current.label} conf={current.confidence:.2f} "
        f"v={current.valence:+.2f} a={current.arousal:+.2f}]"
    )


def format_scene_hint(caption: Optional[str]) -> Optional[str]:
    if not caption or not caption.strip():
        return None
    return f"[scene: {caption.strip()}]"


def format_memory_hint(memories: Iterable[str], max_chars: int = 240) -> Optional[str]:
    mems = [m.strip() for m in memories if m and m.strip()]
    if not mems:
        return None
    joined = " | ".join(mems)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1] + "…"
    return f"[remembered: {joined}]"


def prepare_user_message(
    text: str,
    *,
    emotion_hint: Optional[str] = None,
    scene_hint: Optional[str] = None,
    memory_hint: Optional[str] = None,
) -> str:
    """Combine hints + user text into the message sent to the LLM."""
    parts: list[str] = []
    for hint in (memory_hint, scene_hint, emotion_hint):
        if hint:
            parts.append(hint)
    parts.append(text.strip())
    return " ".join(parts)
