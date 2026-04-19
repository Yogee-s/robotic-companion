"""Per-turn structured tracing.

A `TurnTrace` records timestamps at each phase of a conversation turn
(speech-end, STT done, first LLM token, first audio out, audio end, ...)
plus routing metadata. `TelemetryRecorder` subscribes to the bus, holds
a ring buffer of the most recent N traces, and persists each one as a
single JSONL line under `logs/traces_<date>.jsonl`.

This module is append-only and has no control authority — failures here
must not degrade the conversation. Log-write errors are swallowed with
a DEBUG log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class TurnTrace:
    """One turn's timeline. Timestamps are epoch-seconds floats; None until stamped."""

    turn_id: str
    t_created: float = field(default_factory=time.time)
    t_vad_start: Optional[float] = None
    t_vad_end: Optional[float] = None
    t_stt_first_partial: Optional[float] = None
    t_stt_final: Optional[float] = None
    t_hints_ready: Optional[float] = None
    t_llm_prefill: Optional[float] = None
    t_llm_first_token: Optional[float] = None
    t_first_audio: Optional[float] = None
    t_audio_end: Optional[float] = None
    t_completed: Optional[float] = None

    transcript: str = ""
    route: str = ""                         # chat | vqa | tool
    tool_name: Optional[str] = None
    emotion_label: Optional[str] = None
    emotion_confidence: float = 0.0
    interrupt_reason: Optional[str] = None
    error_class: Optional[str] = None
    cancelled: bool = False

    # ── fluent stamping ─────────────────────────────────────────────────
    def mark(self, phase: str) -> None:
        """Set `t_<phase>` to now() if the attribute exists."""
        attr = f"t_{phase}"
        if hasattr(self, attr) and getattr(self, attr) is None:
            setattr(self, attr, time.time())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryRecorder:
    """Collects TurnTrace objects, persists each as JSONL, and exposes stats."""

    def __init__(self, log_dir: str, ring_size: int = 100) -> None:
        self._log_dir = log_dir
        self._ring: deque[TurnTrace] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        os.makedirs(self._log_dir, exist_ok=True)

    # ── api ─────────────────────────────────────────────────────────────
    def record(self, trace: TurnTrace) -> None:
        """Persist a completed turn (called once per turn, at TurnCompleted)."""
        with self._lock:
            self._ring.append(trace)
        self._write_jsonl(trace)

    def recent(self) -> list[TurnTrace]:
        with self._lock:
            return list(self._ring)

    def latency_ms(self, from_phase: str, to_phase: str) -> list[float]:
        """Differences (ms) between two phase timestamps across recent turns."""
        out: list[float] = []
        with self._lock:
            for tr in self._ring:
                a = getattr(tr, f"t_{from_phase}", None)
                b = getattr(tr, f"t_{to_phase}", None)
                if a is not None and b is not None and b >= a:
                    out.append((b - a) * 1000.0)
        return out

    def percentile(self, values: list[float], pct: float) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
        return s[idx]

    # ── internals ───────────────────────────────────────────────────────
    def _write_jsonl(self, trace: TurnTrace) -> None:
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            path = os.path.join(self._log_dir, f"traces_{date_str}.jsonl")
            with open(path, "a") as fh:
                fh.write(json.dumps(trace.as_dict(), default=str) + "\n")
        except Exception as exc:
            log.debug("telemetry write failed: %r", exc)
