"""Background scene captioning loop.

Runs the VLM at `scene_watch_hz` (default 1.0) on the current camera
frame, publishing a rolling `SceneState` that the conversation manager
and proactive engine can read cheaply. Produces a single sentence every
second or so — never blocks conversation because it runs in its own
thread and holds its own serialised lock on the VLM.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from companion.core.events import Signal
from companion.vision.vlm import MoondreamVLM

log = logging.getLogger(__name__)


@dataclass
class SceneState:
    caption: str = ""
    timestamp: float = 0.0
    latency_ms: float = 0.0


class SceneWatcher:
    def __init__(
        self,
        vlm: MoondreamVLM,
        frame_provider,
        watch_hz: float = 1.0,
    ) -> None:
        """`frame_provider` is a callable that returns the current BGR frame
        or None — typically `emotion_pipeline.get_state().frame`."""
        self.vlm = vlm
        self.frame_provider = frame_provider
        self.watch_hz = float(watch_hz)
        self.on_scene = Signal("scene")
        self._state = SceneState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running or not self.vlm.available:
            if not self.vlm.available:
                log.info("SceneWatcher skipped (VLM unavailable).")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"SceneWatcher started at {self.watch_hz:.1f} Hz")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_state(self) -> SceneState:
        with self._lock:
            return self._state

    def _loop(self) -> None:
        period = 1.0 / max(0.1, self.watch_hz)
        while self._running:
            t0 = time.perf_counter()
            frame = self.frame_provider()
            if isinstance(frame, np.ndarray) and frame.size:
                caption = self.vlm.caption(frame)
                if caption:
                    latency = (time.perf_counter() - t0) * 1000
                    state = SceneState(caption=caption, timestamp=time.time(), latency_ms=latency)
                    with self._lock:
                        self._state = state
                    self.on_scene.emit(state)
            # Sleep the remainder of the period — don't starve the VLM's
            # independent cadence regardless of inference latency.
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
