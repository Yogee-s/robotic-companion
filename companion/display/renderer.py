"""Display renderer interface + factory.

Two backends implement the same `Renderer` protocol:
  - `pygame` — HDMI / dev window (full frame drawn locally)
  - `esp32_serial` — Diymore ESP32 2.8" touchscreen (ESP32 renders, we
    send state commands over /dev/ttyUSB0)

Conversation code never touches pygame or pyserial directly — it sends
`FaceState` objects to the renderer and subscribes to its action callback.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol

from companion.core.config import DisplayConfig
from companion.display.state import FaceState

log = logging.getLogger(__name__)

ActionCallback = Callable[[str, dict], None]


class Renderer(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_face(self, state: FaceState) -> None: ...
    def push_visemes(self, events: list, sample_rate: int) -> None: ...
    def set_action_callback(self, cb: ActionCallback) -> None: ...


def make_renderer(cfg: DisplayConfig) -> Optional[Renderer]:
    backend = cfg.backend.lower()
    if backend == "esp32_serial":
        try:
            from companion.display.backends.esp32_serial import ESP32SerialRenderer

            return ESP32SerialRenderer(cfg)
        except Exception as exc:
            log.warning(f"ESP32 serial backend unavailable: {exc!r}; trying pygame.")
    try:
        from companion.display.backends.pygame import PygameRenderer

        return PygameRenderer(cfg)
    except Exception as exc:
        log.error(f"No display backend available: {exc!r}")
        return None
