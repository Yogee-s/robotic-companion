"""Priority lock around GPU-bound work.

The Jetson Orin Nano has one GPU. Both the main LLM (Gemma) and the
background scene-captioning VLM (also Gemma, multimodal) try to use it.
If the scene captioner grabs the GPU while the LLM is streaming, tokens
stutter and the user hears it.

This arbiter serializes access with priority: realtime callers (LLM
during an active turn, VQA) take precedence; background callers yield
before starting inference and check periodically whether a realtime
caller is waiting.

Usage
-----
    arbiter = GPUArbiter()

    # Main LLM (realtime)
    with arbiter.realtime():
        for tok in llm.generate(...):
            ...

    # Scene captioner (background)
    with arbiter.background() as guard:
        if guard.should_yield():
            return  # skip this tick entirely
        caption = vlm.caption(frame)
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class _BackgroundGuard:
    """Handle returned by `background()`; lets caller check for preemption."""

    def __init__(self, arbiter: "GPUArbiter") -> None:
        self._arb = arbiter

    def should_yield(self) -> bool:
        return self._arb._realtime_waiting.is_set()


class GPUArbiter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._realtime_waiting = threading.Event()
        self._bg_active = 0
        self._bg_cond = threading.Condition()

    # ── realtime path ───────────────────────────────────────────────────
    class _RealtimeCtx:
        def __init__(self, arb: "GPUArbiter") -> None:
            self._arb = arb

        def __enter__(self) -> "GPUArbiter._RealtimeCtx":
            self._arb._realtime_waiting.set()
            self._arb._lock.acquire()
            return self

        def __exit__(self, *_exc) -> None:
            self._arb._realtime_waiting.clear()
            self._arb._lock.release()

    def realtime(self) -> "_RealtimeCtx":
        """Acquire exclusive GPU for a realtime caller."""
        return GPUArbiter._RealtimeCtx(self)

    # ── background path ─────────────────────────────────────────────────
    class _BackgroundCtx:
        def __init__(self, arb: "GPUArbiter") -> None:
            self._arb = arb
            self._guard: _BackgroundGuard = _BackgroundGuard(arb)

        def __enter__(self) -> _BackgroundGuard:
            # Wait until no realtime holder is active. We take the same
            # lock to provide mutual exclusion but release it before
            # inference starts — the guard.should_yield() check is the
            # cooperative mechanism for the actual preemption.
            self._arb._lock.acquire()
            try:
                with self._arb._bg_cond:
                    self._arb._bg_active += 1
            finally:
                self._arb._lock.release()
            return self._guard

        def __exit__(self, *_exc) -> None:
            with self._arb._bg_cond:
                self._arb._bg_active = max(0, self._arb._bg_active - 1)
                self._arb._bg_cond.notify_all()

    def background(self) -> "_BackgroundCtx":
        """Acquire GPU for a background task; realtime always pre-empts."""
        return GPUArbiter._BackgroundCtx(self)
