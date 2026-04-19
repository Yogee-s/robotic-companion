"""Typed error taxonomy.

Every subsystem that can fail in a user-visible way raises one of these
instead of a bare Exception. ConversationManager catches by type and
emits a specific, helpful message via TTS; HealthMonitor tracks error
rates per subsystem.
"""

from __future__ import annotations


class CompanionError(Exception):
    """Base class for all subsystem errors surfaced to the user."""


class STTError(CompanionError):
    """Speech-to-text failure (Parakeet / Whisper)."""


class LLMError(CompanionError):
    """Language model failure (generation hung, load failed, OOM)."""


class TTSError(CompanionError):
    """Text-to-speech failure (Kokoro / Piper / aplay)."""


class ToolError(CompanionError):
    """Tool invocation failure (timer, weather, etc.)."""


class ToolNetworkError(ToolError):
    """Tool failed because a network resource was unreachable."""


class MotorError(CompanionError):
    """Motor subsystem failure (stall, over-temperature, serial drop)."""


class VisionError(CompanionError):
    """Vision pipeline failure (camera unplugged, model crash)."""


class SerialError(CompanionError):
    """Serial-device failure (ESP32 display, motor bus)."""
