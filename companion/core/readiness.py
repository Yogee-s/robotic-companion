"""Startup readiness probe.

Run before `main()` enters its run loop. If anything listed here fails,
we print a structured report to stderr and exit 1 — much better than
booting a half-dead robot that fails silently when the user tries to
use a feature.

Checks:
* Every model file referenced by config.yaml exists.
* (If enabled) the motor serial port is present.
* (If enabled) the ESP32 serial port is present.
* The log directory is writable.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)


@dataclass
class ReadinessItem:
    name: str
    ok: bool
    detail: str = ""
    # Required items must pass or boot aborts. Optional items print as
    # WARN on failure — the subsystem gracefully self-disables.
    required: bool = True


@dataclass
class ReadinessReport:
    items: list[ReadinessItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items if item.required)

    @property
    def has_warnings(self) -> bool:
        return any(not item.ok and not item.required for item in self.items)

    def print(self, stream=sys.stderr) -> None:
        print("─ Readiness probe " + "─" * 42, file=stream)
        for item in self.items:
            if item.ok:
                mark = "OK  "
            elif item.required:
                mark = "FAIL"
            else:
                mark = "WARN"
            detail = f"  ({item.detail})" if item.detail else ""
            print(f"  [{mark}] {item.name}{detail}", file=stream)
        status = "READY" if self.ok else "NOT READY"
        if self.ok and self.has_warnings:
            status = "READY (with warnings)"
        print(f"─ {status} " + "─" * max(0, 58 - len(status)), file=stream)


def _file_ok(path: str, label: str, required: bool = True) -> ReadinessItem:
    if not path:
        return ReadinessItem(name=label, ok=True, detail="(not configured)", required=required)
    if os.path.exists(path):
        return ReadinessItem(name=label, ok=True, detail=path, required=required)
    return ReadinessItem(
        name=label, ok=False, detail=f"missing: {path}", required=required
    )


def _dir_writable(path: str, label: str, required: bool = True) -> ReadinessItem:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".writable_probe")
        with open(probe, "w") as fh:
            fh.write("")
        os.remove(probe)
        return ReadinessItem(name=label, ok=True, detail=path, required=required)
    except Exception as exc:
        return ReadinessItem(
            name=label, ok=False, detail=f"{path}: {exc!r}", required=required
        )


def check_all(cfg) -> ReadinessReport:
    rep = ReadinessReport()

    # ── LLM (required — no conversation without it) ─────────────────────
    try:
        llm_path = cfg.llm_model_path()
    except Exception as exc:
        rep.items.append(
            ReadinessItem(name="LLM model key", ok=False, detail=repr(exc), required=True)
        )
        llm_path = ""
    rep.items.append(_file_ok(llm_path, "LLM model", required=True))

    # Multimodal vision projector (optional — LLM still works text-only)
    if cfg.llm.mmproj_path:
        rep.items.append(
            _file_ok(
                cfg.abspath(cfg.llm.mmproj_path),
                "LLM vision projector (optional)",
                required=False,
            )
        )

    # ── STT (required — can't do anything without it) ───────────────────
    if cfg.stt.backend == "parakeet":
        rep.items.append(
            _file_ok(
                cfg.abspath(cfg.stt.parakeet_model_dir),
                "Parakeet STT model dir",
                required=True,
            )
        )

    # ── EOU / speaker ID (optional — subsystems self-disable) ──────────
    if cfg.eou.enabled:
        rep.items.append(
            _file_ok(cfg.abspath(cfg.eou.model_path), "EOU model (optional)", required=False)
        )
    if cfg.speaker_id.enabled:
        rep.items.append(
            _file_ok(
                cfg.abspath(cfg.speaker_id.model_path),
                "Speaker ID model (optional)",
                required=False,
            )
        )

    # ── Vision (required when vision.enabled) ──────────────────────────
    if cfg.vision.enabled:
        rep.items.append(
            _file_ok(
                cfg.abspath(cfg.vision.yolo_pose_model_path),
                "YOLO pose model",
                required=True,
            )
        )
        rep.items.append(
            _file_ok(
                cfg.abspath(cfg.vision.emotion_model_path),
                "Emotion classifier model",
                required=True,
            )
        )

    # ── FunctionGemma (optional — tool routing gracefully off) ─────────
    if cfg.function_gemma.enabled:
        rep.items.append(
            _file_ok(
                cfg.abspath(cfg.function_gemma.model_path),
                "FunctionGemma model (optional)",
                required=False,
            )
        )

    # ── Motor serial (only check port existence, not torque handshake) ──
    if cfg.motor.enabled and not cfg.motor.sim_only:
        item = (
            ReadinessItem(name="Motor serial port", ok=True, detail=cfg.motor.port)
            if os.path.exists(cfg.motor.port)
            else ReadinessItem(
                name="Motor serial port",
                ok=False,
                detail=f"missing: {cfg.motor.port}",
            )
        )
        rep.items.append(item)

    # ── ESP32 display serial ────────────────────────────────────────────
    if cfg.display.backend == "esp32_serial":
        item = (
            ReadinessItem(name="ESP32 serial port", ok=True, detail=cfg.display.serial_port)
            if os.path.exists(cfg.display.serial_port)
            else ReadinessItem(
                name="ESP32 serial port",
                ok=False,
                detail=f"missing: {cfg.display.serial_port}",
            )
        )
        rep.items.append(item)

    # ── Log directory writable ─────────────────────────────────────────
    rep.items.append(_dir_writable(cfg.abspath(cfg.app.log_dir), "Log directory"))
    rep.items.append(
        _dir_writable(cfg.abspath(cfg.conversation.log_directory), "Conversation logs")
    )
    rep.items.append(_dir_writable(cfg.abspath(cfg.memory.chroma_dir), "Memory DB dir"))

    return rep


def gate(cfg) -> None:
    """Run the probe; exit 1 on failure with a report."""
    rep = check_all(cfg)
    rep.print()
    if not rep.ok:
        raise SystemExit(1)
