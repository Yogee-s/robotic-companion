"""Tool registry — decorator-based registration with auto JSON-Schema.

Usage inside a tool module:

    from companion.tools.registry import tool

    @tool("timer", "Set a countdown timer.")
    def start_timer(duration_seconds: int) -> str:
        ...

The registry walks `companion.tools` at startup and exposes the schemas
to `FunctionGemma` so the LLM knows what arguments each tool expects.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    schema: dict


_TOOLS: dict[str, ToolSpec] = {}


def tool(name: str, description: str) -> Callable:
    """Decorator that registers a function as a callable tool."""
    def decorator(func: Callable) -> Callable:
        schema = _schema_from_signature(name, description, func)
        _TOOLS[name] = ToolSpec(name=name, description=description, func=func, schema=schema)
        return func

    return decorator


def _schema_from_signature(name: str, description: str, func: Callable) -> dict:
    sig = inspect.signature(func)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        json_type = _py_type_to_json(p.annotation)
        props[pname] = {"type": json_type, "description": pname.replace("_", " ")}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
        },
    }


def _py_type_to_json(annotation: Any) -> str:
    if annotation in (int,):
        return "integer"
    if annotation in (float,):
        return "number"
    if annotation in (bool,):
        return "boolean"
    return "string"


def all_schemas() -> list[dict]:
    return [t.schema for t in _TOOLS.values()]


def invoke(name: str, args: dict) -> str:
    t = _TOOLS.get(name)
    if t is None:
        return f"[tool error: unknown tool '{name}']"
    try:
        result = t.func(**args)
    except TypeError as exc:
        return f"[tool error: bad args for '{name}': {exc}]"
    except Exception as exc:
        log.exception(f"Tool '{name}' raised")
        return f"[tool error: {name}: {exc}]"
    return str(result) if result is not None else ""


def load_all_tools() -> list[str]:
    """Import each starter tool so its @tool decorators execute."""
    from companion.tools import (  # noqa: F401
        remind_me,
        stopwatch,
        time_weather,
        timer,
        volume,
    )
    return list(_TOOLS.keys())
