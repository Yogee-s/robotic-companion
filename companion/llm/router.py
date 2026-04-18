"""Decides how to handle each user turn.

Routing order:

  1. **Tool** — if FunctionGemma detects a tool call with high confidence,
     execute it and short-circuit to spoken result.
  2. **VQA** — if the utterance looks vision-dependent ("what is this?",
     "what do you see?", "read this label"), hand the turn + current
     camera frame to the VLM (Moondream) and stream its answer to TTS.
  3. **Chat** — otherwise the main conversation LLM (Gemma 4 E2B) handles
     the reply with emotion/scene/memory context injected.

Each route returns the same shape so the conversation manager can pipe
the reply into TTS uniformly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class Route(Enum):
    CHAT = "chat"
    VQA = "vqa"
    TOOL = "tool"


_VISION_TRIGGERS = re.compile(
    r"\b("
    r"what(?:'s| is)(?: this| that| here)?"
    r"|can you see"
    r"|what do you see"
    r"|describe (?:this|the scene|what|it)"
    r"|read (?:this|the sign|the label|what)"
    r"|look at"
    r"|am i holding"
    r"|how do i look"
    r"|what am i (?:holding|wearing|doing)"
    r"|show me"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    route: Route
    reason: str


def decide_route(user_utterance: str, has_tool_call: bool = False) -> RouteDecision:
    if has_tool_call:
        return RouteDecision(Route.TOOL, "FunctionGemma detected a tool call")
    if _VISION_TRIGGERS.search(user_utterance):
        return RouteDecision(Route.VQA, "vision-referential phrase detected")
    return RouteDecision(Route.CHAT, "default chat route")
