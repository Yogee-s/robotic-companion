#!/usr/bin/env python3
"""Go / no-go startup self-check.

Walks every subsystem the companion depends on and prints a concise
green/yellow/red table. Useful both from the CLI (`python3 scripts/verify.py`)
and from `main.py` as a startup banner.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


from companion.core.config import load_config  # noqa: E402


GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def row(status: str, label: str, detail: str = "") -> None:
    colour = {"ok": GREEN, "warn": YELLOW, "err": RED}.get(status, RESET)
    tag = {"ok": "✓", "warn": "!", "err": "✗"}.get(status, "?")
    print(f"  {colour}{tag}{RESET}  {label:<28}  {DIM}{detail}{RESET}")


def main() -> int:
    cfg = load_config(str(_ROOT / "config.yaml"))
    print(f"\nCompanion self-check — project at {cfg.project_root}\n")

    # Runtime packages
    try:
        import torch  # type: ignore

        row("ok" if torch.cuda.is_available() else "warn",
            "torch",
            f"{torch.__version__} CUDA={torch.cuda.is_available()}")
    except ImportError:
        row("err", "torch", "not installed")

    try:
        import llama_cpp  # type: ignore
        row("ok", "llama-cpp-python", getattr(llama_cpp, "__version__", "?"))
    except ImportError:
        row("err", "llama-cpp-python", "run scripts/setup.sh")

    try:
        import onnxruntime as ort  # type: ignore
        providers = ort.get_available_providers()
        row("ok" if "CUDAExecutionProvider" in providers else "warn",
            "onnxruntime", ",".join(providers))
    except ImportError:
        row("err", "onnxruntime", "not installed")

    for mod, name in (("faster_whisper", "faster-whisper"),
                      ("kokoro_onnx", "kokoro-onnx"),
                      ("pygame", "pygame"), ("serial", "pyserial"),
                      ("PyQt5", "PyQt5"), ("pysbd", "pysbd"),
                      ("mem0", "mem0ai")):
        try:
            __import__(mod)
            row("ok", name)
        except ImportError:
            row("warn", name, "optional / install via requirements.txt")

    # Models on disk
    print("\n  Models")
    for rel in (
        cfg.llm.model_paths[cfg.llm.model],
        cfg.vlm.model_path,
        cfg.stt.parakeet_model_dir + "/encoder.onnx",
        cfg.vision.face_model_path,
        cfg.vision.emotion_model_path,
        cfg.eou.model_path,
        cfg.speaker_id.model_path,
        "models/kokoro/kokoro-v1.0.fp16.onnx",
    ):
        abs_path = cfg.abspath(rel)
        row("ok" if os.path.exists(abs_path) else "warn", rel,
            f"{os.path.getsize(abs_path) / 1e6:.1f} MB" if os.path.exists(abs_path) else "missing")

    # Hardware
    print("\n  Hardware")
    row("ok" if os.path.exists(cfg.display.serial_port) else "warn",
        "screen serial", cfg.display.serial_port)
    row("ok" if os.path.exists("/dev/video0") else "warn",
        "camera /dev/video0", "")

    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
