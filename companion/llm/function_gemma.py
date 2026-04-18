"""FunctionGemma-270M — a tiny sidecar LLM that parses user turns into tool calls.

Loaded via llama-cpp-python alongside the main chat LLM. Before each user
turn is sent to the chat model, FunctionGemma is asked whether the turn
should trigger a tool. If yes and its confidence exceeds the threshold,
we execute the tool and feed its result into the chat LLM as a system
note, instead of sending the user turn raw.

Tool schemas come from the `@tool` decorator in `companion/tools/registry.py`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: dict
    confidence: float


class FunctionGemma:
    def __init__(
        self,
        model_path: str,
        enabled: bool = True,
        confidence_threshold: float = 0.55,
    ) -> None:
        self.model_path = model_path
        self.enabled = enabled
        self.confidence_threshold = float(confidence_threshold)
        self._llm = None
        self._tool_schemas_json = "[]"
        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError:
            log.warning("llama-cpp-python not installed — FunctionGemma disabled.")
            self.enabled = False
            return
        import os
        if not os.path.exists(self.model_path):
            log.warning(f"FunctionGemma model not found at {self.model_path}; disabled.")
            self.enabled = False
            return
        try:
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1,
                n_ctx=1024,
                n_batch=256,
                verbose=False,
            )
            log.info(f"FunctionGemma-270M loaded from {self.model_path}")
        except Exception as exc:
            log.warning(f"FunctionGemma load failed: {exc!r}")
            self.enabled = False

    @property
    def available(self) -> bool:
        return self.enabled and self._llm is not None

    def set_tools(self, schemas: list[dict]) -> None:
        """Register the callable tools' JSON schemas."""
        self._tool_schemas_json = json.dumps(schemas, indent=2)

    def detect(self, user_utterance: str) -> Optional[ToolCall]:
        if not self.available:
            return None
        prompt = self._build_prompt(user_utterance)
        try:
            assert self._llm is not None
            out = self._llm.create_completion(
                prompt=prompt,
                max_tokens=128,
                temperature=0.0,
                top_p=0.9,
                stop=["</tool_call>", "\nUser:"],
            )
            text = out["choices"][0]["text"].strip()
        except Exception as exc:
            log.debug(f"FunctionGemma inference failed: {exc!r}")
            return None
        return self._parse(text)

    def _build_prompt(self, user: str) -> str:
        return (
            "You are a tool-calling model. Given a user message and a list of tool "
            "schemas, decide whether a tool should be called. If yes, output a "
            "<tool_call> JSON block; if no, output exactly NO_TOOL.\n\n"
            f"Tools:\n{self._tool_schemas_json}\n\n"
            f"User: {user}\n\n"
            "Response:"
        )

    def _parse(self, text: str) -> Optional[ToolCall]:
        if "NO_TOOL" in text.upper():
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments") or obj.get("args") or {}
        confidence = float(obj.get("confidence", 0.7))
        if not name or confidence < self.confidence_threshold:
            return None
        return ToolCall(name=name, args=args, confidence=confidence)
