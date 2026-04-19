"""BehaviorEngine — the 20 Hz coordinator for motor + face display.

Replaces the old `_face_loop` in main.py. A single thread ticks at
`behavior_tick_hz` and:

* reads the latest `EmotionState` from the vision pipeline,
* reads the DOA angle from the ReSpeaker (if connected),
* tracks the current conversation state (updated via bus subscription),
* applies the per-state `GainSchedule` to the `FaceTracker`,
* derives a `FaceState` from emotion + conversation state + DOA and
  pushes it to the `Renderer`.

The `FaceTracker` runs its own async control loop at ~15 Hz; this engine
only adjusts its gains and (if motors are disabled) acts as the display-
only driver.

Affect tags from the LLM fire one-shot expression overlays via the
bus subscription.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from companion.behavior.tracking import GainSchedule
from companion.conversation.states import ConversationState
from companion.core.events import (
    AffectTag,
    FaceAppeared,
    FaceLost,
    StateChanged,
    VisemeStream,
)
from companion.display.expressions import emotion_to_face
from companion.display.state import ConversationalState, FaceState

log = logging.getLogger(__name__)


# Map conversation state → renderer's ConversationalState (they overlap
# but aren't identical — our machine has CAPTURING_INTENT and RECOVERING
# which the renderer collapses to thinking / idle).
_CONV_MAP: dict[str, ConversationalState] = {
    ConversationState.IDLE_WATCHING:    ConversationalState.IDLE,
    ConversationState.LISTENING:        ConversationalState.LISTENING,
    ConversationState.CAPTURING_INTENT: ConversationalState.THINKING,
    ConversationState.THINKING:         ConversationalState.THINKING,
    ConversationState.SPEAKING:         ConversationalState.SPEAKING,
    ConversationState.RECOVERING:       ConversationalState.IDLE,
}

# Map affect tag → renderer expression overlay. The firmware already
# supports "confused" | "surprised" | "excited" | "angry" | "sad"; we
# translate the richer LLM palette onto that limited set.
_AFFECT_EXPRESSION: dict[str, str] = {
    "happy":        "excited",
    "curious":      "confused",
    "confused":     "confused",
    "surprised":    "surprised",
    "affectionate": "excited",
    "angry":        "angry",
    "sad":          "sad",
}

_AFFECT_HOLD_SEC = 1.2


class BehaviorEngine:
    def __init__(
        self,
        *,
        renderer,
        emotion_pipeline=None,
        respeaker=None,
        face_tracker=None,
        event_bus=None,
        tick_hz: float = 20.0,
    ) -> None:
        self._renderer = renderer
        self._emotion = emotion_pipeline
        self._respeaker = respeaker
        self._tracker = face_tracker
        self._bus = event_bus
        self._period = 1.0 / max(1.0, float(tick_hz))

        self._gain = GainSchedule(face_tracker) if face_tracker is not None else None

        self._conv_state: str = ConversationState.IDLE_WATCHING
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Affect-tag one-shot
        self._affect_until: float = 0.0
        self._affect_expr: Optional[str] = None

        # Face-presence edge detection — used to publish FaceAppeared /
        # FaceLost on the bus so the Coordinator can react without
        # every module polling the vision pipeline.
        self._prev_has_face: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._bus is not None:
            self._bus.subscribe(StateChanged, self._on_state_changed)
            self._bus.subscribe(AffectTag, self._on_affect_tag)
            self._bus.subscribe(VisemeStream, self._on_viseme_stream)
        if self._tracker is not None:
            try:
                self._tracker.start_async()
            except Exception as exc:
                log.warning("FaceTracker start failed: %r", exc)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="behavior-engine"
        )
        self._thread.start()
        log.info("BehaviorEngine started")

    def stop(self, timeout_s: float = 2.0) -> None:
        self._running = False
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
        log.info("BehaviorEngine stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── event subscribers ────────────────────────────────────────────────
    def _on_state_changed(self, ev: StateChanged) -> None:
        with self._state_lock:
            self._conv_state = ev.new
        if self._gain is not None:
            self._gain.apply(ev.new)

    def _on_affect_tag(self, ev: AffectTag) -> None:
        expr = _AFFECT_EXPRESSION.get(ev.tag.lower())
        if expr is None:
            return
        self._affect_expr = expr
        self._affect_until = time.time() + _AFFECT_HOLD_SEC

    def _on_viseme_stream(self, ev: VisemeStream) -> None:
        if self._renderer is None:
            return
        push = getattr(self._renderer, "push_visemes", None)
        if push is None:
            return
        try:
            push(ev.events, ev.sample_rate)
        except Exception as exc:
            log.debug("push_visemes failed: %r", exc)

    # ── main tick ────────────────────────────────────────────────────────
    def _loop(self) -> None:
        from companion.vision.pipeline import EmotionState

        while self._running:
            t0 = time.perf_counter()
            try:
                if self._emotion is not None:
                    em = self._emotion.get_state()
                else:
                    em = EmotionState()

                # Edge-publish face presence transitions for the Coordinator.
                has_face = bool(getattr(em, "has_face", False))
                if has_face != self._prev_has_face and self._bus is not None:
                    if has_face:
                        bbox = getattr(em, "bbox", None)
                        if bbox is not None:
                            self._bus.publish(
                                FaceAppeared(
                                    bbox=tuple(int(v) for v in bbox),
                                    timestamp=time.time(),
                                )
                            )
                    else:
                        self._bus.publish(FaceLost(timestamp=time.time()))
                self._prev_has_face = has_face

                doa: Optional[float] = None
                if self._respeaker is not None and getattr(self._respeaker, "is_connected", False):
                    try:
                        doa = float(self._respeaker.get_doa())
                    except Exception:
                        doa = None

                with self._state_lock:
                    conv_s = self._conv_state
                conv_enum = _CONV_MAP.get(conv_s, ConversationalState.IDLE)

                fs = emotion_to_face(em, conv_enum, doa_angle_deg=doa)

                # Affect-tag overlay takes precedence for a short window.
                if self._affect_expr and time.time() < self._affect_until:
                    fs.expression = self._affect_expr
                elif self._affect_expr and time.time() >= self._affect_until:
                    self._affect_expr = None

                if self._renderer is not None:
                    self._renderer.set_face(fs)

            except Exception as exc:
                log.debug("BehaviorEngine tick error: %r", exc)

            elapsed = time.perf_counter() - t0
            rem = self._period - elapsed
            if rem > 0:
                time.sleep(rem)
