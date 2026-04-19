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


def _draw_spiral_eye(pygame_mod, surf, cx: int, cy: int, radius: int,
                     colour=(220, 220, 240), width: int = 2) -> None:
    """Cartoon confusion-swirl eye: a single continuous spiral traced
    from centre outward, so it reads unambiguously as a vortex."""
    # Filled backdrop disk so the spiral reads on the face bg.
    pygame_mod.draw.circle(surf, (18, 18, 30), (cx, cy), radius + 2)
    # Archimedean spiral: r(t) = k*t, ~2 full turns.
    pts = []
    turns = 2.1
    steps = 34
    for i in range(steps + 1):
        frac = i / steps
        theta = frac * turns * 2 * math.pi
        r = frac * radius
        pts.append((cx + int(r * math.cos(theta)),
                    cy + int(r * math.sin(theta))))
    if len(pts) > 1:
        pygame_mod.draw.lines(surf, colour, False, pts, width)
    # Centre dot for contrast
    pygame_mod.draw.circle(surf, colour, (cx, cy), 2)


def _draw_dizzy_swirl(pygame_mod, surf, x: int, y: int, size: int = 28) -> None:
    """Big hovering 'dizzy' swirl — the classic cartoon confusion symbol.

    Single line tracing an Archimedean spiral outward, thicker than the
    eye spirals so it reads from across the room.
    """
    col = (250, 230, 170)          # warm yellow
    shadow = (180, 160, 100)       # faint drop shadow for depth
    turns = 2.4
    steps = 54
    pts = []
    for i in range(steps + 1):
        frac = i / steps
        theta = frac * turns * 2 * math.pi
        r = frac * size
        pts.append((x + int(r * math.cos(theta)),
                    y + int(r * math.sin(theta))))
    # Shadow first, then the bright swirl on top
    shadow_pts = [(px + 1, py + 1) for (px, py) in pts]
    if len(pts) > 1:
        pygame_mod.draw.lines(surf, shadow, False, shadow_pts, 4)
        pygame_mod.draw.lines(surf, col, False, pts, 3)
    # Small end-cap dot so the spiral has a clear 'start'
    pygame_mod.draw.circle(surf, col, pts[-1], 3)


def _draw_tear(pygame_mod, surf, x: int, y: int) -> None:
    """A single teardrop — used for sad expression."""
    col = (110, 170, 240)
    # Body (drop shape): triangle on top, circle at bottom
    pygame_mod.draw.polygon(surf, col, [(x, y), (x - 5, y + 9), (x + 5, y + 9)])
    pygame_mod.draw.circle(surf, col, (x, y + 12), 5)
    # Highlight
    pygame_mod.draw.circle(surf, (220, 235, 255), (x - 1, y + 10), 1)


def _draw_sparkle(pygame_mod, surf, x: int, y: int, sz: int = 8) -> None:
    """Four-point sparkle — used for excited."""
    col = (255, 240, 150)
    # Horizontal + vertical strokes
    pygame_mod.draw.line(surf, col, (x - sz, y), (x + sz, y), 2)
    pygame_mod.draw.line(surf, col, (x, y - sz), (x, y + sz), 2)
    # Diagonals, half-length, for a plus-with-star look
    d = int(sz * 0.55)
    pygame_mod.draw.line(surf, col, (x - d, y - d), (x + d, y + d), 1)
    pygame_mod.draw.line(surf, col, (x - d, y + d), (x + d, y - d), 1)


def _draw_anger_mark(pygame_mod, surf, x: int, y: int) -> None:
    """Four radial 'veins' — the classic anime anger symbol."""
    col = (230, 90, 90)
    for ang in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        dx = int(10 * math.cos(ang))
        dy = int(10 * math.sin(ang))
        pygame_mod.draw.line(surf, col, (x, y), (x + dx, y + dy), 3)
    # Small hub
    pygame_mod.draw.circle(surf, col, (x, y), 3)


def _draw_wavy_mouth(pygame_mod, surf, cx: int, cy: int, width: int) -> None:
    """Zig-zag mouth — used for confused (can't decide a smile or a frown)."""
    col = (240, 200, 210)
    pts = []
    steps = 6
    for i in range(steps + 1):
        x = cx - width // 2 + int(width * (i / steps))
        y = cy + (4 if i % 2 else -4)
        pts.append((x, y))
    pygame_mod.draw.lines(surf, col, False, pts, 3)

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
        brow_col = (240, 240, 255)
        mouth_col = (240, 200, 210)

        # ── Dedicated sleep face ────────────────────────────────────────
        if fs.sleep:
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.arc(
                    surf, eye_col,
                    pygame_mod.Rect(ex - eye_rad, eye_y - 4, eye_rad * 2, 8),
                    0, math.pi, 3,
                )
            mouth_y = int(cy + h * 0.14)
            mw = int(w * 0.08)
            pygame_mod.draw.arc(
                surf, mouth_col,
                pygame_mod.Rect(cx - mw, mouth_y - 4, 2 * mw, 10),
                math.pi, 2 * math.pi, 3,
            )
            _draw_zzz(pygame_mod, surf, cx + int(w * 0.22), int(cy - h * 0.20))
            return

        # ── Dedicated confused face — spiral eyes + wavy mouth + dizzy swirl
        if fs.expression == "confused":
            # Spiral swirl eyes (override normal eye drawing).
            swirl_eye_rad = int(eye_rad * 1.4)
            for side in (-1, 1):
                ex = cx + side * eye_dx
                _draw_spiral_eye(pygame_mod, surf, ex, eye_y,
                                 swirl_eye_rad, width=2)
            # Asymmetric brows — left up-slope, right down-slope (classic
            # "one-eyebrow-raised" confused look).
            brow_y = eye_y - swirl_eye_rad - 8
            pygame_mod.draw.line(
                surf, brow_col,
                (cx - eye_dx - swirl_eye_rad, brow_y - 4),
                (cx - eye_dx + swirl_eye_rad, brow_y + 8),
                3,
            )
            pygame_mod.draw.line(
                surf, brow_col,
                (cx + eye_dx - swirl_eye_rad, brow_y + 8),
                (cx + eye_dx + swirl_eye_rad, brow_y - 4),
                3,
            )
            # Zigzag mouth
            mouth_y = int(cy + h * 0.14)
            _draw_wavy_mouth(pygame_mod, surf, cx, mouth_y, int(w * 0.20))
            # Big dizzy swirl hovering above the head — the defining marker.
            swirl_size = max(22, int(min(w, h) * 0.11))
            _draw_dizzy_swirl(
                pygame_mod, surf,
                cx + int(w * 0.24),
                int(cy - h * 0.28),
                size=swirl_size,
            )
            return

        # ── Base eyes ───────────────────────────────────────────────────
        for side in (-1, 1):
            ex = cx + side * eye_dx + eye_offset
            if blink_closed:
                pygame_mod.draw.line(surf, eye_col, (ex - eye_rad, eye_y),
                                     (ex + eye_rad, eye_y), 4)
            else:
                pygame_mod.draw.circle(surf, eye_col, (ex, eye_y), eye_rad)
                # Pupil size — big for surprise (wide-eyed small-pupil look),
                # normal otherwise. Excited gets a highlight gleam.
                pupil_rad = eye_rad // 2
                if fs.expression == "surprised":
                    pupil_rad = max(2, eye_rad // 3)
                pygame_mod.draw.circle(
                    surf, pupil_col,
                    (ex + eye_offset // 2, eye_y), pupil_rad,
                )
                if fs.expression == "excited":
                    pygame_mod.draw.circle(
                        surf, (255, 255, 255),
                        (ex + eye_offset // 2 - pupil_rad // 2,
                         eye_y - pupil_rad // 2),
                        max(1, pupil_rad // 3),
                    )

        # ── Brows ───────────────────────────────────────────────────────
        # Shape varies per expression.
        brow_y0 = eye_y - eye_rad - 8
        if fs.expression == "angry":
            # Sharp V, inner-corners very low, converging toward nose.
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.line(
                    surf, brow_col,
                    (ex - eye_rad, brow_y0 - 6),              # outer high
                    (ex + side * eye_rad, brow_y0 + 10),       # inner low
                    4,
                )
        elif fs.expression == "sad":
            # Inner corners raised (classic sad brow).
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.line(
                    surf, brow_col,
                    (ex - eye_rad, brow_y0 + 4),               # outer low
                    (ex + side * eye_rad, brow_y0 - 8),         # inner high
                    3,
                )
        elif fs.expression == "surprised":
            # Both brows lifted high and flat.
            brow_y0 -= 14
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.line(
                    surf, brow_col,
                    (ex - eye_rad, brow_y0),
                    (ex + eye_rad, brow_y0),
                    3,
                )
        else:
            # V/A-driven default (neutral + excited fall here).
            brow_tilt = int(fs.valence * -18) + int(fs.arousal * 6)
            brow_y_lift = int(max(0.0, fs.arousal) * 6)
            for side in (-1, 1):
                ex = cx + side * eye_dx
                y_off = side * brow_tilt
                y0 = eye_y - eye_rad - 8 - brow_y_lift
                pygame_mod.draw.line(
                    surf, brow_col,
                    (ex - eye_rad, y0 + y_off),
                    (ex + eye_rad, y0 - y_off),
                    3,
                )

        # ── Mouth ───────────────────────────────────────────────────────
        mouth_y = int(cy + h * 0.14)
        mouth_w = int(w * 0.26)
        viseme_h = {
            "rest": 6, "mm": 6, "eh": 14,
            "oh": 28, "ahh": 36, "ee": 10,
            "fv": 8, "l": 18,
        }.get(viseme, 10)
        arousal_open = max(0, int(max(0.0, fs.arousal) * 18))
        mouth_h = viseme_h + arousal_open
        smile = fs.valence * 18

        if fs.expression == "surprised":
            # Big round 'O' — the defining surprise shape.
            r = int(min(w, h) * 0.06)
            pygame_mod.draw.circle(surf, mouth_col, (cx, mouth_y), r, 3)
        elif fs.expression == "excited":
            # Big filled grin with teeth.
            grin_rect = pygame_mod.Rect(
                cx - int(mouth_w * 0.55),
                mouth_y - 8 - int(smile),
                int(mouth_w * 1.1),
                22 + int(abs(smile)),
            )
            pygame_mod.draw.arc(surf, mouth_col, grin_rect,
                                math.pi, 2 * math.pi, 5)
            pygame_mod.draw.line(
                surf, mouth_col,
                (grin_rect.left + 10, grin_rect.centery),
                (grin_rect.right - 10, grin_rect.centery),
                2,
            )
        elif fs.expression == "angry":
            # Gritted horizontal teeth — tight line.
            pygame_mod.draw.line(
                surf, mouth_col,
                (cx - mouth_w // 2, mouth_y), (cx + mouth_w // 2, mouth_y),
                4,
            )
            # Vertical teeth separators
            for k in range(-2, 3):
                x = cx + k * int(mouth_w * 0.12)
                pygame_mod.draw.line(surf, mouth_col,
                                     (x, mouth_y - 5), (x, mouth_y + 5), 2)
        elif fs.expression == "sad":
            # Downturned arc (flip the smile).
            frown_h = 14
            rect = pygame_mod.Rect(
                cx - mouth_w // 2,
                mouth_y - frown_h // 2,
                mouth_w,
                frown_h,
            )
            pygame_mod.draw.arc(surf, mouth_col, rect, 0, math.pi, 4)
        else:
            rect = pygame_mod.Rect(
                cx - mouth_w // 2,
                mouth_y - mouth_h // 2 - int(smile),
                mouth_w,
                mouth_h + int(abs(smile)),
            )
            pygame_mod.draw.arc(surf, mouth_col, rect, math.pi, 2 * math.pi, 4)

        # ── Ornaments — drawn on top of the base face ───────────────────
        if fs.expression == "excited":
            # Three sparkles around the head
            for sx, sy, sz in (
                (cx - int(w * 0.28), int(cy - h * 0.28), 9),
                (cx + int(w * 0.28), int(cy - h * 0.22), 11),
                (cx + int(w * 0.20), int(cy + h * 0.02),  7),
            ):
                _draw_sparkle(pygame_mod, surf, sx, sy, sz)
        elif fs.expression == "angry":
            _draw_anger_mark(
                pygame_mod, surf,
                cx - int(w * 0.22), int(cy - h * 0.22),
            )
        elif fs.expression == "sad":
            # One teardrop under the left eye
            _draw_tear(
                pygame_mod, surf,
                cx - eye_dx + 3, eye_y + eye_rad + 2,
            )

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
