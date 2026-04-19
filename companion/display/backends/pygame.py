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


def _draw_question_mark(pygame_mod, surf, x: int, y: int,
                        sz: int = 22, col=(250, 230, 170)) -> None:
    """Floating '?' glyph drawn from a hook arc + stem + dot. No font
    dependency — just lines and arcs so it renders the same everywhere."""
    # Hook of the '?'
    rect = pygame_mod.Rect(x - sz // 2, y, sz, sz)
    pygame_mod.draw.arc(surf, col, rect, math.radians(-30), math.pi, 3)
    # Vertical stem down from the hook
    pygame_mod.draw.line(surf, col,
                         (x, y + sz - 4), (x, y + int(sz * 1.35)), 3)
    # Dot below
    pygame_mod.draw.circle(surf, col, (x, y + int(sz * 1.55)), 3)


def _draw_gear(pygame_mod, surf, cx: int, cy: int, r: int = 12) -> None:
    """Small cog — the universal 'thinking / processing' icon."""
    body = (200, 200, 220)
    hole = (18, 18, 30)
    pygame_mod.draw.circle(surf, body, (cx, cy), r, 2)
    for i in range(8):
        ang = i * math.pi / 4
        ox = int((r + 3) * math.cos(ang))
        oy = int((r + 3) * math.sin(ang))
        pygame_mod.draw.circle(surf, body, (cx + ox, cy + oy), 2)
    pygame_mod.draw.circle(surf, hole, (cx, cy), max(2, r // 3))


def _draw_ellipsis(pygame_mod, surf, x: int, y: int,
                   dot_sp: int = 9) -> None:
    """Three horizontal dots '...' for the thinking face."""
    col = (220, 220, 240)
    for i in range(3):
        pygame_mod.draw.circle(surf, col, (x + i * dot_sp, y), 2)


def _draw_bulb(pygame_mod, surf, x: int, y: int, sz: int = 22) -> None:
    """Classic lightbulb with rays — the 'idea!' icon."""
    bulb = (250, 230, 90)
    bulb_outline = (140, 110, 30)
    ray_col = (235, 210, 90)
    base_col = (150, 130, 70)
    # Radiating rays
    for i in range(8):
        ang = i * math.pi / 4
        ri = int(sz * 0.78)
        ro = int(sz * 1.15)
        pygame_mod.draw.line(
            surf, ray_col,
            (x + int(ri * math.cos(ang)), y + int(ri * math.sin(ang))),
            (x + int(ro * math.cos(ang)), y + int(ro * math.sin(ang))),
            2,
        )
    # Bulb body
    r_body = int(sz * 0.55)
    pygame_mod.draw.circle(surf, bulb, (x, y), r_body)
    pygame_mod.draw.circle(surf, bulb_outline, (x, y), r_body, 1)
    # Metal base + thread lines
    base_y = y + int(r_body * 0.85)
    pygame_mod.draw.rect(
        surf, base_col,
        pygame_mod.Rect(x - r_body // 2, base_y, r_body, 5),
    )
    for i in range(2):
        yy = base_y + 2 + i * 2
        pygame_mod.draw.line(surf, bulb_outline,
                             (x - r_body // 2, yy),
                             (x + r_body // 2, yy), 1)


def _draw_heart(pygame_mod, surf, cx: int, cy: int,
                size: int = 10, col=(240, 120, 160)) -> None:
    """Small filled heart — two top circles + triangle bottom."""
    r = max(2, size // 2)
    pygame_mod.draw.circle(surf, col, (cx - r + 1, cy - r // 2), r)
    pygame_mod.draw.circle(surf, col, (cx + r - 1, cy - r // 2), r)
    pts = [
        (cx - r - 1, cy - 1),
        (cx + r + 1, cy - 1),
        (cx, cy + size),
    ]
    pygame_mod.draw.polygon(surf, col, pts)


def _draw_sound_arcs(pygame_mod, surf, cx: int, cy: int,
                     n: int = 3, spacing: int = 8,
                     facing: int = 1) -> None:
    """Concentric quarter-arcs indicating inbound sound (listening icon).
    `facing=+1` → arcs open to the right; `-1` → open to the left."""
    col = (130, 200, 240)
    # Quarter-arc angular span
    if facing > 0:
        a0, a1 = -math.pi / 4, math.pi / 4
    else:
        a0, a1 = math.pi * 3 / 4, math.pi * 5 / 4
    for i in range(1, n + 1):
        r = i * spacing
        rect = pygame_mod.Rect(cx - r, cy - r, 2 * r, 2 * r)
        pygame_mod.draw.arc(surf, col, rect, a0, a1, 2)


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

        # ── Dedicated confused face: swirly eyes + wavy mouth + ?? around head
        if fs.expression == "confused":
            # Swirly eyes, prominently sized so the vortex is obvious
            swirl_eye_rad = int(eye_rad * 1.5)
            for side in (-1, 1):
                ex = cx + side * eye_dx
                _draw_spiral_eye(pygame_mod, surf, ex, eye_y,
                                 swirl_eye_rad, width=2)
            # Asymmetric brows — classic "one raised, one lowered" look
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
            # Zigzag mouth (wavering between smile and frown)
            mouth_y = int(cy + h * 0.14)
            _draw_wavy_mouth(pygame_mod, surf, cx, mouth_y, int(w * 0.20))
            # Three '?' marks floating around the head at varying sizes
            # and warmth-shifted colours so they read as a constellation,
            # not tiled wallpaper.
            q_sz = max(16, int(min(w, h) * 0.08))
            marks = (
                # (x_offset, y_offset, scale, colour)
                (-int(w * 0.30), -int(h * 0.28), 1.15, (250, 230, 170)),
                (+int(w * 0.28), -int(h * 0.24), 1.00, (255, 210, 140)),
                (+int(w * 0.36), +int(h * 0.02), 0.70, (220, 200, 150)),
            )
            for dx, dy, scale, col in marks:
                _draw_question_mark(
                    pygame_mod, surf,
                    cx + dx, cy + dy,
                    sz=int(q_sz * scale),
                    col=col,
                )
            return

        # ── Thinking: pupils up-and-right + '...' + cog ─────────────
        if fs.expression == "thinking":
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.circle(surf, eye_col, (ex, eye_y), eye_rad)
                # Pupil shifted up-and-slightly-right (gaze into thought)
                pup_dx = int(eye_rad * 0.35)
                pup_dy = -int(eye_rad * 0.45)
                pygame_mod.draw.circle(
                    surf, pupil_col,
                    (ex + pup_dx, eye_y + pup_dy),
                    max(3, eye_rad // 2),
                )
            # One brow up (quizzical, asymmetric)
            brow_y = eye_y - eye_rad - 10
            pygame_mod.draw.line(
                surf, brow_col,
                (cx - eye_dx - eye_rad, brow_y + 4),
                (cx - eye_dx + eye_rad, brow_y - 2),
                3,
            )
            pygame_mod.draw.line(
                surf, brow_col,
                (cx + eye_dx - eye_rad, brow_y - 6),     # outer high
                (cx + eye_dx + eye_rad, brow_y - 12),    # inner higher
                3,
            )
            # Neutral closed mouth line
            mouth_y = int(cy + h * 0.14)
            mw = int(w * 0.16)
            pygame_mod.draw.line(surf, mouth_col,
                                 (cx - mw // 2, mouth_y),
                                 (cx + mw // 2, mouth_y), 3)
            # Ornaments: '...' over the head, and a gear to the upper-right
            _draw_ellipsis(
                pygame_mod, surf,
                cx - int(w * 0.06), int(cy - h * 0.32),
                dot_sp=max(8, int(min(w, h) * 0.035)),
            )
            _draw_gear(
                pygame_mod, surf,
                cx + int(w * 0.26), int(cy - h * 0.28),
                r=max(10, int(min(w, h) * 0.05)),
            )
            return

        # ── Idea: bright iris + tiny pupil + lightbulb + grin ────────
        if fs.expression == "idea":
            for side in (-1, 1):
                ex = cx + side * eye_dx
                # Big white iris with tiny pupil — "aha!" wide-eyed look
                pygame_mod.draw.circle(surf, eye_col, (ex, eye_y),
                                       eye_rad + 1)
                pygame_mod.draw.circle(surf, pupil_col, (ex, eye_y),
                                       max(2, eye_rad // 3))
                # Specular highlight
                pygame_mod.draw.circle(
                    surf, (255, 255, 255),
                    (ex - eye_rad // 3, eye_y - eye_rad // 3),
                    max(1, eye_rad // 4),
                )
            # Brows high and arched
            brow_y = eye_y - eye_rad - 14
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.arc(
                    surf, brow_col,
                    pygame_mod.Rect(ex - eye_rad, brow_y - 3,
                                    eye_rad * 2, 10),
                    0, math.pi, 3,
                )
            # Big grin
            mouth_y = int(cy + h * 0.14)
            rect = pygame_mod.Rect(
                cx - int(w * 0.14), mouth_y - 8,
                int(w * 0.28), 24,
            )
            pygame_mod.draw.arc(surf, mouth_col, rect,
                                math.pi, 2 * math.pi, 5)
            # The lightbulb above the head — the defining icon
            _draw_bulb(
                pygame_mod, surf,
                cx, int(cy - h * 0.32),
                sz=max(22, int(min(w, h) * 0.14)),
            )
            return

        # ── Love: heart eyes + floating hearts + soft smile ──────────
        if fs.expression == "love":
            heart_col = (240, 120, 160)
            heart_size = int(eye_rad * 2.2)
            for side in (-1, 1):
                ex = cx + side * eye_dx
                _draw_heart(pygame_mod, surf, ex, eye_y,
                            size=heart_size, col=heart_col)
            # Gentle relaxed brows
            brow_y = eye_y - eye_rad - 12
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.line(
                    surf, brow_col,
                    (ex - eye_rad, brow_y),
                    (ex + eye_rad, brow_y),
                    2,
                )
            # Soft smile
            mouth_y = int(cy + h * 0.14)
            rect = pygame_mod.Rect(
                cx - int(w * 0.10), mouth_y - 4,
                int(w * 0.20), 14,
            )
            pygame_mod.draw.arc(surf, mouth_col, rect,
                                math.pi, 2 * math.pi, 3)
            # Little hearts floating around the head at varied sizes
            for dx, dy, scale in (
                (-int(w * 0.28), -int(h * 0.28), 0.9),
                (+int(w * 0.30), -int(h * 0.22), 1.1),
                (+int(w * 0.22), +int(h * 0.02), 0.7),
                (-int(w * 0.32), -int(h * 0.08), 0.6),
            ):
                _draw_heart(pygame_mod, surf,
                            cx + dx, cy + dy,
                            size=max(6, int(12 * scale)),
                            col=heart_col)
            return

        # ── Wink: one eye closed, one gleaming, cocky smile ──────────
        if fs.expression == "wink":
            # Left eye closed (curved lid), right eye open with gleam.
            # (Swap the signs if you prefer the opposite eye.)
            left_x = cx - eye_dx
            right_x = cx + eye_dx
            # Closed left: smile-shaped arc
            pygame_mod.draw.arc(
                surf, eye_col,
                pygame_mod.Rect(left_x - eye_rad, eye_y - eye_rad // 2,
                                eye_rad * 2, eye_rad),
                math.pi, 2 * math.pi, 3,
            )
            # Open right
            pygame_mod.draw.circle(surf, eye_col, (right_x, eye_y), eye_rad)
            pygame_mod.draw.circle(surf, pupil_col, (right_x, eye_y),
                                   eye_rad // 2)
            pygame_mod.draw.circle(
                surf, (255, 255, 255),
                (right_x - eye_rad // 3, eye_y - eye_rad // 3),
                max(1, eye_rad // 4),
            )
            # Brows: neutral on closed side, slightly raised on open
            brow_y = eye_y - eye_rad - 10
            pygame_mod.draw.line(
                surf, brow_col,
                (left_x - eye_rad, brow_y + 2),
                (left_x + eye_rad, brow_y + 2), 3,
            )
            pygame_mod.draw.line(
                surf, brow_col,
                (right_x - eye_rad, brow_y - 4),
                (right_x + eye_rad, brow_y), 3,
            )
            # Lopsided smirk: higher on the open-eye side
            mouth_y = int(cy + h * 0.14)
            pts = [
                (cx - int(w * 0.12), mouth_y + 4),
                (cx, mouth_y + 2),
                (cx + int(w * 0.12), mouth_y - 4),
            ]
            pygame_mod.draw.lines(surf, mouth_col, False, pts, 3)
            return

        # ── Listening: normal eyes + inbound sound arcs ──────────────
        if fs.expression == "listening":
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.circle(surf, eye_col, (ex, eye_y), eye_rad)
                pygame_mod.draw.circle(surf, pupil_col, (ex, eye_y),
                                       eye_rad // 2)
            # Attentive flat brows, slightly raised
            brow_y = eye_y - eye_rad - 10
            for side in (-1, 1):
                ex = cx + side * eye_dx
                pygame_mod.draw.line(
                    surf, brow_col,
                    (ex - eye_rad, brow_y),
                    (ex + eye_rad, brow_y),
                    3,
                )
            # Small neutral-interested mouth: short flat line
            mouth_y = int(cy + h * 0.14)
            mw = int(w * 0.12)
            pygame_mod.draw.line(surf, mouth_col,
                                 (cx - mw // 2, mouth_y),
                                 (cx + mw // 2, mouth_y), 3)
            # Sound arcs on BOTH sides so it reads as "hearing from around"
            _draw_sound_arcs(pygame_mod, surf,
                             cx + int(w * 0.34), cy,
                             n=3, spacing=max(6, int(min(w, h) * 0.035)),
                             facing=+1)
            _draw_sound_arcs(pygame_mod, surf,
                             cx - int(w * 0.34), cy,
                             n=3, spacing=max(6, int(min(w, h) * 0.035)),
                             facing=-1)
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
