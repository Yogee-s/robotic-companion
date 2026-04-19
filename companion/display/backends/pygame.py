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


def _draw_zzz(pygame_mod, surf, x: int, y: int) -> None:
    """Draw three ascending Z's near (x, y) to indicate sleep.
    Uses line strokes so no font is required."""
    col = (220, 220, 240)
    for i, scale in enumerate((1.0, 0.7, 0.45)):
        sz = int(18 * scale)
        ox = x - i * int(sz * 0.9)
        oy = y - i * int(sz * 1.1)
        # Z shape: top, diagonal, bottom
        pygame_mod.draw.line(surf, col, (ox, oy), (ox + sz, oy), 3)
        pygame_mod.draw.line(surf, col, (ox + sz, oy), (ox, oy + sz), 3)
        pygame_mod.draw.line(surf, col, (ox, oy + sz), (ox + sz, oy + sz), 3)

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

        # Arousal scales the eye size (wide-eyed when excited, narrow when tired).
        # Range: 0.7× (arousal=-1) → 1.35× (arousal=+1).
        eye_scale = 1.0 + max(-1.0, min(1.0, fs.arousal)) * 0.35
        base_eye_rad = int(min(w, h) * 0.06)
        eye_rad = max(4, int(base_eye_rad * eye_scale))

        eye_y = int(cy - h * 0.08)
        eye_dx = int(w * 0.15)
        eye_offset = int(fs.gaze_x * eye_rad * 0.6)
        eye_col = (220, 220, 240)
        pupil_col = (20, 24, 40)

        # Dedicated sleep face: closed eyes as gentle curves, no brows, small m-mouth, zzz glyph.
        if fs.sleep:
            for side in (-1, 1):
                ex = cx + side * eye_dx
                # Curved eyelid: two short segments approximating a smile-shaped lid
                pygame_mod.draw.arc(
                    surf, eye_col,
                    pygame_mod.Rect(ex - eye_rad, eye_y - 4, eye_rad * 2, 8),
                    0, math.pi, 3,
                )
            # Small 'm' mouth
            mouth_y = int(cy + h * 0.14)
            mw = int(w * 0.08)
            pygame_mod.draw.arc(
                surf, (240, 200, 210),
                pygame_mod.Rect(cx - mw, mouth_y - 4, 2 * mw, 10),
                math.pi, 2 * math.pi, 3,
            )
            _draw_zzz(pygame_mod, surf, cx + int(w * 0.22), int(cy - h * 0.20))
            return

        # Eyes
        for side in (-1, 1):
            ex = cx + side * eye_dx + eye_offset
            if blink_closed:
                pygame_mod.draw.line(surf, eye_col, (ex - eye_rad, eye_y),
                                     (ex + eye_rad, eye_y), 4)
            else:
                pygame_mod.draw.circle(surf, eye_col, (ex, eye_y), eye_rad)
                pygame_mod.draw.circle(surf, pupil_col,
                                       (ex + eye_offset // 2, eye_y), eye_rad // 2)

        # Eyebrows — tilt with valence (down-inner for happy, up-inner for sad/angry)
        # Raise with arousal (higher brows = more alert). Amplified.
        brow_tilt = int(fs.valence * -18) + int(fs.arousal * 6)
        brow_y_lift = int(max(0.0, fs.arousal) * 6)
        for side in (-1, 1):
            ex = cx + side * eye_dx
            y_off = side * brow_tilt
            y0 = eye_y - eye_rad - 8 - brow_y_lift
            pygame_mod.draw.line(
                surf, (240, 240, 255),
                (ex - eye_rad, y0 + y_off),
                (ex + eye_rad, y0 - y_off),
                3,
            )

        # Mouth — valence curves it; arousal opens it; viseme shapes during speech.
        mouth_y = int(cy + h * 0.14)
        mouth_w = int(w * 0.26)
        viseme_h = {
            "rest": 6, "mm": 6, "eh": 14,
            "oh": 28, "ahh": 36, "ee": 10,
            "fv": 8, "l": 18,
        }.get(viseme, 10)
        # Arousal opens the mouth (wide excited grin, flat calm mouth).
        arousal_open = max(0, int(max(0.0, fs.arousal) * 18))
        mouth_h = viseme_h + arousal_open
        smile = fs.valence * 18
        rect = pygame_mod.Rect(
            cx - mouth_w // 2,
            mouth_y - mouth_h // 2 - int(smile),
            mouth_w,
            mouth_h + int(abs(smile)),
        )
        # Draw lower half as arc, but if aroused+happy, fill a wider smile.
        if fs.valence > 0.4 and fs.arousal > 0.4:
            # Big excited grin — filled arc
            pygame_mod.draw.arc(surf, (240, 200, 210), rect, math.pi, 2 * math.pi, 5)
            # Show teeth hint
            pygame_mod.draw.line(
                surf, (240, 200, 210),
                (rect.left + 8, rect.centery),
                (rect.right - 8, rect.centery),
                2,
            )
        else:
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
