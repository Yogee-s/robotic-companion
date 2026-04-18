"""Proactive engagement — the robot occasionally starts the conversation.

Background loop that watches emotion history, scene captions, speaker ID,
and time-of-day. If a rule triggers (familiar face arrives; sadness
persists; scene transitions from empty → person), it speaks an opener
through the conversation manager.

Heavy rate-limiting so the robot isn't pestering. Opt-in via
`app.proactive_enabled: true`.
"""

from __future__ import annotations

import datetime as _dt
import logging
import random
import threading
import time
from typing import Optional

from companion.core.config import AppConfig
from companion.vision.pipeline import EmotionPipeline
from companion.vision.scene_watcher import SceneWatcher

log = logging.getLogger(__name__)


class ProactiveEngine:
    MIN_INTERVAL_S = 90.0  # at most one opener every 90 s

    def __init__(
        self,
        cfg: AppConfig,
        conversation,
        emotion: Optional[EmotionPipeline] = None,
        scene: Optional[SceneWatcher] = None,
    ) -> None:
        self.cfg = cfg
        self.conversation = conversation
        self.emotion = emotion
        self.scene = scene
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_fire = 0.0
        self._saw_face_last = False
        self._sad_since: Optional[float] = None

    def start(self) -> None:
        if not self.cfg.app.proactive_enabled:
            log.info("Proactive engine disabled in config.")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Proactive engine started")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            time.sleep(5.0)
            try:
                self._tick()
            except Exception:
                log.exception("Proactive tick failed")

    def _tick(self) -> None:
        now = time.time()
        if self.emotion is None or now - self._last_fire < self.MIN_INTERVAL_S:
            return
        state = self.emotion.get_state()

        # Rule 1 — arrival: no face → face in the last 2 s
        if state.has_face and not self._saw_face_last:
            self._fire("Oh, hey! Good to see you.")
            self._saw_face_last = True
            return
        if not state.has_face:
            self._saw_face_last = False
            self._sad_since = None
            return

        # Rule 2 — sustained sadness
        if state.valence < -0.4:
            if self._sad_since is None:
                self._sad_since = now
            elif now - self._sad_since > 120.0:
                self._fire("You've been a little quiet. Anything you want to share?")
                self._sad_since = None
                return
        else:
            self._sad_since = None

        # Rule 3 — morning greeting
        hour = _dt.datetime.now().hour
        if 6 <= hour < 10 and not self._history_has_said_morning_today():
            self._fire(random.choice([
                "Morning! How'd you sleep?",
                "Hey, good morning. What's the plan today?",
            ]))

    def _history_has_said_morning_today(self) -> bool:
        today = _dt.date.today().isoformat()
        return any(
            "morning" in (e.get("content") or "").lower()
            and str(today) in (e.get("content") or "")
            for e in self.conversation.get_history()
        )

    def _fire(self, utterance: str) -> None:
        self._last_fire = time.time()
        log.info(f"Proactive: {utterance!r}")
        # Speak directly through the conversation manager's TTS path.
        try:
            self.conversation._speak_text(utterance)
        except Exception:
            log.exception("Proactive speak failed")
