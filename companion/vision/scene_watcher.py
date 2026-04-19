"""Background scene captioning loop.

Runs the multimodal LLM at `scene_watch_hz` (default 0.5) on the current
camera frame, publishing a rolling `SceneState` that the conversation
manager and proactive engine can read cheaply. Produces a single
sentence every couple of seconds — never blocks conversation because:

* it yields to the `GPUArbiter` whenever a realtime LLM turn is pending,
* it pauses entirely while a conversation turn is active (listening,
  thinking, or speaking),
* it runs in its own thread behind its own lock.

Consolidation note: the dedicated Moondream VLM was dropped in favor of
the multimodal Gemma already loaded for chat. The caller passes the
`LLMEngine` in; its `caption(frame, question)` method gives us the same
captioning behavior using one model.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from companion.core.events import Signal

log = logging.getLogger(__name__)


@dataclass
class SceneState:
    caption: str = ""
    timestamp: float = 0.0
    latency_ms: float = 0.0


class SceneWatcher:
    """Periodic scene captioning on the current camera frame."""

    SCENE_QUESTION = "Describe the scene in one short sentence."

    def __init__(
        self,
        llm,
        frame_provider: Callable[[], Optional[np.ndarray]],
        watch_hz: float = 0.5,
        *,
        arbiter=None,            # optional GPUArbiter
        is_turn_active: Callable[[], bool] = lambda: False,
    ) -> None:
        """
        Args
        ----
        llm : LLMEngine
            Must expose `.caption(frame, question)` and `.is_multimodal`.
        frame_provider : callable
            Returns the current BGR frame (or None). Usually wired to
            `emotion_pipeline.get_state().frame`.
        watch_hz : float
            Captioning cadence target. Actual rate is lower whenever
            inference is slow or an active turn is pausing us.
        arbiter : GPUArbiter | None
            If provided, each caption runs under `arbiter.background()`
            so realtime LLM turns preempt us.
        is_turn_active : callable
            Returns True while a conversation turn is in flight; when
            True, we skip captioning this tick entirely.
        """
        self.llm = llm
        self.frame_provider = frame_provider
        self.watch_hz = float(watch_hz)
        self.on_scene = Signal("scene")
        self._arbiter = arbiter
        self._is_turn_active = is_turn_active
        self._state = SceneState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        if not getattr(self.llm, "is_multimodal", False):
            log.info("SceneWatcher skipped (LLM is not multimodal).")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scene-watcher")
        self._thread.start()
        log.info("SceneWatcher started at %.2f Hz", self.watch_hz)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── state read ───────────────────────────────────────────────────────
    def get_state(self) -> SceneState:
        with self._lock:
            return self._state

    # ── main loop ────────────────────────────────────────────────────────
    def _loop(self) -> None:
        period = 1.0 / max(0.1, self.watch_hz)
        while self._running:
            t0 = time.perf_counter()
            if self._is_turn_active():
                # Skip — the LLM is busy serving a user turn.
                time.sleep(period)
                continue

            frame = self.frame_provider()
            caption: Optional[str] = None
            if isinstance(frame, np.ndarray) and frame.size:
                caption = self._caption_with_arbiter(frame)

            if caption:
                latency = (time.perf_counter() - t0) * 1000
                state = SceneState(
                    caption=caption, timestamp=time.time(), latency_ms=latency
                )
                with self._lock:
                    self._state = state
                self.on_scene.emit(state)

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _caption_with_arbiter(self, frame: np.ndarray) -> Optional[str]:
        if self._arbiter is None:
            return self.llm.caption(frame, self.SCENE_QUESTION)
        with self._arbiter.background() as guard:
            if guard.should_yield():
                return None
            return self.llm.caption(frame, self.SCENE_QUESTION)
