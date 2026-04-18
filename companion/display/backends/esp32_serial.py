"""ESP32 serial renderer — sends face state commands to the Diymore 2.8"
touchscreen over /dev/ttyUSB0 and receives button/touch events back.

Protocol (line-delimited ASCII):

  Jetson → ESP32
    FACE v=+0.72 a=+0.30 talk=1 listen=0 think=0 sleep=0 blink=0 gaze=-12
    VISEME ahh                      # during TTS playback
    SCENE face | quickgrid | morelist
    PRIVACY 1 | 0

  ESP32 → Jetson
    BTN mute_mic | stop_talking | sleep | more | timer | remind_me | ...
    TOUCH x y                       # raw touch (for debugging)

Baud 115200. Frame rate ~30 Hz on face updates, on demand for others.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from companion.core.config import DisplayConfig
from companion.display.lip_sync import VisemeEvent
from companion.display.state import FaceState, Scene

log = logging.getLogger(__name__)


class ESP32SerialRenderer:
    def __init__(self, cfg: DisplayConfig) -> None:
        self.cfg = cfg
        self._port: Optional["serial.Serial"] = None  # type: ignore[name-defined]
        self._state = FaceState()
        self._state_lock = threading.Lock()
        self._visemes: list[VisemeEvent] = []
        self._visemes_lock = threading.Lock()
        self._viseme_started_at: Optional[float] = None
        self._last_viseme_sent: Optional[str] = None
        self._action_cb: Optional[Callable[[str, dict], None]] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyserial not installed") from exc
        self._port = serial.Serial(
            self.cfg.serial_port,
            self.cfg.serial_baud,
            timeout=0.05,
            write_timeout=0.5,
        )
        # Give the ESP32 a moment to reset after DTR toggles on port open.
        time.sleep(0.5)
        self._running = True
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx_thread.start()
        self._rx_thread.start()
        log.info(f"ESP32SerialRenderer connected on {self.cfg.serial_port}")

    def stop(self) -> None:
        self._running = False
        for t in (self._tx_thread, self._rx_thread):
            if t is not None:
                t.join(timeout=2.0)
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass

    def set_face(self, state: FaceState) -> None:
        with self._state_lock:
            self._state = state

    def push_visemes(self, events: list, sample_rate: int) -> None:
        with self._visemes_lock:
            self._visemes = list(events)
            self._viseme_started_at = time.time()
            self._last_viseme_sent = None

    def set_action_callback(self, cb: Callable[[str, dict], None]) -> None:
        self._action_cb = cb

    # ── TX loop: stream face state + visemes at 30 Hz ───────────────────
    def _tx_loop(self) -> None:
        period = 1.0 / 30.0
        while self._running:
            t0 = time.perf_counter()
            with self._state_lock:
                fs = self._state
            self._send_face(fs)
            self._send_viseme(time.time())
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _send_face(self, fs: FaceState) -> None:
        line = (
            f"FACE v={fs.valence:+.2f} a={fs.arousal:+.2f} "
            f"talk={int(fs.talking)} listen={int(fs.listening)} "
            f"think={int(fs.thinking)} sleep={int(fs.sleep)} "
            f"gaze={fs.gaze_x * 45:+.0f} privacy={int(fs.privacy)}\n"
        )
        self._write(line)

    def _send_viseme(self, now: float) -> None:
        with self._visemes_lock:
            events = list(self._visemes)
            start = self._viseme_started_at
        if not events or start is None:
            return
        elapsed = now - start
        current = "rest"
        for ev in events:
            if ev.start_s <= elapsed:
                current = ev.viseme
            else:
                break
        if current != self._last_viseme_sent:
            self._write(f"VISEME {current}\n")
            self._last_viseme_sent = current

    def _write(self, line: str) -> None:
        if self._port is None:
            return
        try:
            self._port.write(line.encode("ascii", errors="ignore"))
        except Exception as exc:
            log.debug(f"ESP32 write failed: {exc!r}")

    # ── RX loop: parse button / touch events ────────────────────────────
    def _rx_loop(self) -> None:
        buf = b""
        while self._running:
            if self._port is None:
                break
            try:
                data = self._port.read(64)
            except Exception as exc:
                log.debug(f"ESP32 read failed: {exc!r}")
                time.sleep(0.1)
                continue
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                self._handle_line(line.decode("ascii", errors="ignore").strip())

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        if line.startswith("BTN "):
            name = line[4:].strip()
            self._fire(name)
        elif line.startswith("TOUCH "):
            parts = line.split()
            try:
                x, y = int(parts[1]), int(parts[2])
                self._fire("touch", x=x, y=y)
            except (IndexError, ValueError):
                pass

    def _fire(self, name: str, **kwargs) -> None:
        if self._action_cb is not None:
            try:
                self._action_cb(name, kwargs)
            except Exception:
                log.exception("ESP32 action callback raised")
