#!/usr/bin/env python3
"""Download every model the companion needs in one idempotent script.

Skips files that already exist and match expected sizes. Safe to re-run —
it only fetches missing pieces.

Active models (always downloaded):
  - LLM:    Gemma 4 E2B + Llama 3.2 1B Q4_K_M (llama.cpp GGUF)
  - STT:    Parakeet-TDT-0.6B-v3 ONNX export (int8)
  - TTS:    Piper hfc_female-medium
  - Vision: YuNet face ONNX + HSEmotion ENet-B0 ONNX

Disabled but downloadable (for testing):
  - VLM:    Moondream-2 Q4 GGUF + mmproj (disabled in config.yaml)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "models"


def _download(url: str, dest: Path, min_mb: int = 0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > min_mb * 1024 * 1024:
        print(f"  ✓ {dest.name} (already present)")
        return True
    print(f"  → {dest.name} ← {url}")
    try:
        urlretrieve(url, dest)
        print(f"  ✓ {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  ✗ {dest.name}  ({exc!r})")
        if dest.exists():
            dest.unlink()
        return False


def _hf_download(repo: str, path: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ {dest.name} (already present)")
        return True
    print(f"  → {dest.name} ← {repo}/{path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached = hf_hub_download(repo_id=repo, filename=path)
        shutil.copy2(cached, dest)
        print(f"  ✓ {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except GatedRepoError:
        print(f"  ✗ {dest.name}  (gated — accept the license at https://huggingface.co/{repo})")
        return False
    except RepositoryNotFoundError:
        print(f"  ✗ {dest.name}  (repo {repo} not found)")
        return False
    except Exception as exc:
        print(f"  ✗ {dest.name}  ({exc!r})")
        return False


def llm() -> None:
    print("── LLM (Gemma 4 E2B + Llama 3.2 1B) ──")
    target = _MODELS / "llm"
    target.mkdir(parents=True, exist_ok=True)
    # Gemma 4 E2B — latest lightweight multimodal model for embedded (Jetson)
    _hf_download(
        "unsloth/gemma-4-E2B-it-GGUF",
        "gemma-4-E2B-it-Q4_K_M.gguf",
        target / "gemma-4-e2b-it-q4_k_m.gguf",
    )
    # Llama 3.2 1B — smaller fallback, proven stable on 8 GB Orin
    _hf_download(
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        target / "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    )


def stt() -> None:
    print("── STT (Parakeet-TDT-0.6B-v3) ──")
    # nvidia/parakeet-tdt-0.6b-v3 ships only .nemo; sherpa-onnx provides the
    # int8 ONNX export. We rename to plain foo.onnx locally to match
    # companion/audio/stt.py's expected layout.
    target = _MODELS / "stt"
    target.mkdir(parents=True, exist_ok=True)
    repo = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    for src, dst in (
        ("encoder.int8.onnx", "encoder.onnx"),
        ("decoder.int8.onnx", "decoder.onnx"),
        ("joiner.int8.onnx", "joiner.onnx"),
        ("tokens.txt", "tokens.txt"),
    ):
        _hf_download(repo, src, target / dst)


def tts() -> None:
    print("── TTS (Piper) ──")
    t = _MODELS / "tts"
    t.mkdir(parents=True, exist_ok=True)
    _hf_download(
        "rhasspy/piper-voices",
        "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
        t / "en_US-hfc_female-medium.onnx",
    )
    _hf_download(
        "rhasspy/piper-voices",
        "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json",
        t / "en_US-hfc_female-medium.onnx.json",
    )


def vision() -> None:
    print("── Vision (YuNet + HSEmotion) ──")
    v = _MODELS / "vision"
    v.mkdir(parents=True, exist_ok=True)
    _download(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        v / "face_detection_yunet_2023mar.onnx",
    )
    _download(
        "https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
        v / "enet_b0_8_best_afew.onnx",
    )


def vlm() -> None:
    print("── VLM (Moondream-2) — disabled by default, for testing ──")
    v = _MODELS / "vlm"
    v.mkdir(parents=True, exist_ok=True)
    repo = "ggml-org/moondream2-20250414-GGUF"
    _hf_download(repo, "moondream2-text-model-f16_ct-vicuna.gguf", v / "moondream2-q4.gguf")
    _hf_download(repo, "moondream2-mmproj-f16-20250414.gguf", v / "moondream2-mmproj-f16.gguf")


def check_espeak() -> None:
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
        print("✓ espeak-ng present (required by Kokoro)")
    except Exception:
        print("✗ espeak-ng missing — install with: sudo apt install espeak-ng")


def main() -> int:
    _MODELS.mkdir(exist_ok=True)
    print(f"Downloading models into {_MODELS}\n")

    print("=== Active models ===")
    llm()
    stt()
    tts()
    vision()
    check_espeak()

    print("\n=== Disabled models (for testing) ===")
    vlm()

    print("\nDone. Re-run this script any time to pick up missing files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
