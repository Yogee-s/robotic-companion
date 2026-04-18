"""Pygame renderer — full face drawn locally on HDMI.

Used in two situations:
 1. Development — no ESP32 attached, face lives on the Jetson's monitor.
 2. Debug GUI "Face" tab — always available, regardless of hardware.

Renders in a background thread, snapshots `FaceState` each frame, and
smoothly interpolates between updates. Supports the QUICK_GRID and
MORE_LIST overlays, cross-fading between scenes. Touch is not expected
here (that's the ESP32 job), but clicks / keys on the dev window still
dispatch actions via the action callback.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Optional

from companion.core.config import DisplayConfig
from companion.display.lip_sync import VisemeEvent
from companion.display.state import (
    MORE_LIST_ACTIONS,
    QUICK_GRID_ACTIONS,
    FaceState,
    Scene,
)

log = logging.getLogger(__name__)


class PygameRenderer:
    def __init__(self, cfg: DisplayConfig) -> None:
        self.cfg = cfg
        self._state = FaceState()
        self._state_lock = threading.Lock()
        self._visemes: list[VisemeEvent] = []
        self._visemes_lock = threading.Lock()
        self._viseme_started_at: Optional[float] = None
        self._action_cb: Optional[Callable[[str, dict], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_interaction = 0.0
        self._scene = Scene.FACE

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"PygameRenderer started ({self.cfg.width}x{self.cfg.height})")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_face(self, state: FaceState) -> None:
        with self._state_lock:
            self._state = state

    def push_visemes(self, events: list, sample_rate: int) -> None:
        with self._visemes_lock:
            self._visemes = list(events)
            self._viseme_started_at = time.time()

    def set_action_callback(self, cb: Callable[[str, dict], None]) -> None:
        self._action_cb = cb

    # ── main loop ────────────────────────────────────────────────────────
    def _loop(self) -> None:
        try:
            import pygame  # type: ignore
        except ImportError:
            log.error("pygame not installed — cannot render face on HDMI.")
            self._running = False
            return

        pygame.init()
        flags = pygame.FULLSCREEN if self.cfg.fullscreen else 0
        screen = pygame.display.set_mode((self.cfg.width, self.cfg.height), flags)
        pygame.display.set_caption("Companion · Face")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 20)

        blink_phase = 0.0
        last_frame_t = time.time()
        while self._running:
            # Event pump — dev-time mouse/keyboard dispatch
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                    self._handle_tap(event, screen.get_size())

            now = time.time()
            dt = now - last_frame_t
            last_frame_t = now
            with self._state_lock:
                fs = self._state

            blink_phase += dt * fs.blink_rate_hz
            blink_closed = (blink_phase % 1.0) > 0.92  # short blink each period

            current_viseme = self._current_viseme(now)

            # ── draw ────────────────────────────────────────────────────
            screen.fill((18, 18, 30))
            self._draw_face(
                pygame,
                screen,
                fs,
                blink_closed=blink_closed or fs.sleep,
                viseme=current_viseme,
            )

            if fs.privacy:
                self._draw_privacy_band(pygame, screen)

            if self._scene == Scene.QUICK_GRID:
                self._draw_quick_grid(pygame, screen, font)
            elif self._scene == Scene.MORE_LIST:
                self._draw_more_list(pygame, screen, font)

            # Auto-dismiss overlay after inactivity
            if (
                self._scene != Scene.FACE
                and (now - self._last_interaction) > self.cfg.auto_dismiss_seconds
            ):
                self._scene = Scene.FACE

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    # ── face drawing ─────────────────────────────────────────────────────
    @staticmethod
    def _draw_face(pygame_mod, surf, fs: FaceState, *, blink_closed: bool, viseme: str) -> None:
        w, h = surf.get_size()
        cx, cy = w // 2, h // 2

        # Eyes
        eye_y = int(cy - h * 0.08)
        eye_dx = int(w * 0.15)
        eye_rad = int(min(w, h) * 0.06)
        eye_offset = int(fs.gaze_x * eye_rad * 0.6)
        for side in (-1, 1):
            ex = cx + side * eye_dx + eye_offset
            if blink_closed:
                pygame_mod.draw.line(surf, (220, 220, 240), (ex - eye_rad, eye_y), (ex + eye_rad, eye_y), 4)
            else:
                pygame_mod.draw.circle(surf, (220, 220, 240), (ex, eye_y), eye_rad)
                pygame_mod.draw.circle(surf, (20, 24, 40), (ex + eye_offset // 2, eye_y), eye_rad // 2)

        # Eyebrows — tilt with emotion
        brow_tilt = int(fs.valence * -10) + int(fs.arousal * 4)
        for side in (-1, 1):
            ex = cx + side * eye_dx
            y_off = side * brow_tilt
            pygame_mod.draw.line(
                surf,
                (240, 240, 255),
                (ex - eye_rad, eye_y - eye_rad - 6 + y_off),
                (ex + eye_rad, eye_y - eye_rad - 6 - y_off),
                3,
            )

        # Mouth — curve with valence, height with viseme
        mouth_y = int(cy + h * 0.14)
        mouth_w = int(w * 0.26)
        mouth_h = {
            "rest": 6, "mm": 6, "eh": 14,
            "oh": 28, "ahh": 36, "ee": 10,
            "fv": 8, "l": 18,
        }.get(viseme, 10)
        smile = fs.valence * 14
        rect = pygame_mod.Rect(
            cx - mouth_w // 2,
            mouth_y - mouth_h // 2 - int(smile),
            mouth_w,
            mouth_h + int(abs(smile)),
        )
        pygame_mod.draw.arc(surf, (240, 200, 210), rect, math.pi, 2 * math.pi, 4)

    @staticmethod
    def _draw_privacy_band(pygame_mod, surf) -> None:
        w, h = surf.get_size()
        band = pygame_mod.Rect(0, int(h * 0.32), w, int(h * 0.16))
        pygame_mod.draw.rect(surf, (20, 22, 30), band)

    # ── scene overlays ───────────────────────────────────────────────────
    def _draw_quick_grid(self, pygame_mod, surf, font) -> None:
        w, h = surf.get_size()
        dim = pygame_mod.Surface((w, h), pygame_mod.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        surf.blit(dim, (0, 0))
        tile_w, tile_h = w // 2, h // 2
        labels = ("mute mic", "stop talking", "sleep", "more")
        for i, label in enumerate(labels):
            r = pygame_mod.Rect((i % 2) * tile_w, (i // 2) * tile_h, tile_w, tile_h)
            pygame_mod.draw.rect(surf, (49, 50, 68), r.inflate(-8, -8), border_radius=12)
            txt = font.render(label, True, (220, 220, 240))
            surf.blit(txt, txt.get_rect(center=r.center))

    def _draw_more_list(self, pygame_mod, surf, font) -> None:
        w, h = surf.get_size()
        dim = pygame_mod.Surface((w, h), pygame_mod.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        surf.blit(dim, (0, 0))
        row_h = 36
        for idx, (_action, label) in enumerate(MORE_LIST_ACTIONS):
            r = pygame_mod.Rect(8, 4 + idx * row_h, w - 16, row_h - 4)
            pygame_mod.draw.rect(surf, (49, 50, 68), r, border_radius=8)
            txt = font.render(label, True, (220, 220, 240))
            surf.blit(txt, (r.x + 12, r.y + 8))

    # ── touch / click handling ──────────────────────────────────────────
    def _handle_tap(self, event, size) -> None:
        w, h = size
        self._last_interaction = time.time()
        if self._scene == Scene.FACE:
            self._scene = Scene.QUICK_GRID
            return
        if self._scene == Scene.QUICK_GRID:
            x, y = event.pos if hasattr(event, "pos") else (event.x * w, event.y * h)
            col = 0 if x < w / 2 else 1
            row = 0 if y < h / 2 else 1
            idx = row * 2 + col
            action = QUICK_GRID_ACTIONS[idx]
            if action == "more":
                self._scene = Scene.MORE_LIST
            else:
                self._scene = Scene.FACE
                self._fire(action)
            return
        if self._scene == Scene.MORE_LIST:
            y = event.pos[1] if hasattr(event, "pos") else event.y * h
            row = int((y - 4) // 36)
            if 0 <= row < len(MORE_LIST_ACTIONS):
                action = MORE_LIST_ACTIONS[row][0]
                self._scene = Scene.FACE
                self._fire(action)
            return

    def _fire(self, name: str, **kwargs) -> None:
        if self._action_cb is not None:
            try:
                self._action_cb(name, kwargs)
            except Exception:
                log.exception("Display action callback raised")

    # ── viseme timeline ─────────────────────────────────────────────────
    def _current_viseme(self, now: float) -> str:
        with self._visemes_lock:
            events = list(self._visemes)
            start = self._viseme_started_at
        if not events or start is None:
            return "rest"
        elapsed = now - start
        current = "rest"
        for ev in events:
            if ev.start_s <= elapsed:
                current = ev.viseme
            else:
                break
        return current
