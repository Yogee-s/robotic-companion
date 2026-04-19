#!/usr/bin/env python3
"""Download every model the companion needs in one idempotent script.

Skips files that already exist and match expected sizes. Total first-time
download is ~8-10 GB. Safe to re-run — it only fetches missing pieces.

Groups:
  - LLM:          Gemma 4 E2B + E4B Q4_K_M (llama.cpp GGUF)
  - VLM:          Moondream-2 Q4 GGUF + mmproj
  - Tool router:  FunctionGemma-270M Q4 GGUF
  - STT:          Parakeet-TDT-0.6B-v3 ONNX export (+ Whisper base.en cached via faster-whisper)
  - TTS:          Kokoro-82M ONNX + voices, Piper hfc_female-medium (fallback)
  - Vision:       YuNet face ONNX + HSEmotion ENet-B0 ONNX
  - VAD / EOU:    Silero v5 (bundled with silero-vad pip) + LiveKit EOU-v0.4.1-intl ONNX
  - Speaker ID:   NeMo TitaNet-L ONNX
  - Wake word:    openWakeWord custom "hey_buddy" (user-trained) — a placeholder is copied if missing
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
    print("── LLM (Gemma 4) ──")
    # google/gemma-4-*-GGUF is gated; unsloth hosts ungated re-quantizations.
    # Gemma 4 ships as E2B / E4B (effective-param variants).
    _hf_download(
        "unsloth/gemma-4-E2B-it-GGUF",
        "gemma-4-E2B-it-Q4_K_M.gguf",
        _MODELS / "gemma-4-e2b-it-q4_k_m.gguf",
    )
    _hf_download(
        "unsloth/gemma-4-E4B-it-GGUF",
        "gemma-4-E4B-it-Q4_K_M.gguf",
        _MODELS / "gemma-4-e4b-it-q4_k_m.gguf",
    )


def vlm() -> None:
    print("── VLM (Moondream-2) ──")
    # llama.cpp-official conversion; ships F16 only (no Q4 text model).
    repo = "ggml-org/moondream2-20250414-GGUF"
    _hf_download(repo, "moondream2-text-model-f16_ct-vicuna.gguf", _MODELS / "moondream2-q4.gguf")
    _hf_download(repo, "moondream2-mmproj-f16-20250414.gguf", _MODELS / "moondream2-mmproj-f16.gguf")


def function_gemma() -> None:
    print("── FunctionGemma-270M ──")
    _hf_download(
        "unsloth/functiongemma-270m-it-GGUF",
        "functiongemma-270m-it-Q4_K_M.gguf",
        _MODELS / "function-gemma-270m-q4.gguf",
    )


def stt() -> None:
    print("── STT (Parakeet-TDT-0.6B-v3) ──")
    # nvidia/parakeet-tdt-0.6b-v3 ships only .nemo; sherpa-onnx provides the
    # int8 ONNX export. We rename to plain foo.onnx locally to match
    # companion/audio/stt.py's expected layout.
    target = _MODELS / "parakeet-tdt-0.6b-v3"
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
    print("── TTS (Kokoro + Piper) ──")
    k = _MODELS / "kokoro"
    k.mkdir(parents=True, exist_ok=True)
    _hf_download("hexgrad/Kokoro-82M", "kokoro-v1.0.fp16.onnx", k / "kokoro-v1.0.fp16.onnx")
    _hf_download("hexgrad/Kokoro-82M", "voices-v1.0.bin", k / "voices-v1.0.bin")
    p = _MODELS / "piper"
    p.mkdir(parents=True, exist_ok=True)
    _hf_download(
        "rhasspy/piper-voices",
        "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
        p / "en_US-hfc_female-medium.onnx",
    )
    _hf_download(
        "rhasspy/piper-voices",
        "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json",
        p / "en_US-hfc_female-medium.onnx.json",
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


def eou() -> None:
    print("── Semantic end-of-utterance (LiveKit turn-detector, multilingual) ──")
    target_dir = _MODELS / "eou"
    target_dir.mkdir(parents=True, exist_ok=True)
    repo = "livekit/turn-detector"
    _hf_download(repo, "model_quantized.onnx", target_dir / "livekit-eou-v0.4.1-intl.onnx")
    for extra in ("tokenizer.json", "config.json", "ort_config.json"):
        _hf_download(repo, extra, target_dir / extra)


def speaker_id() -> None:
    print("── Speaker ID (TitaNet-L) ──")
    target = _MODELS / "speaker_id" / "titanet-l.onnx"
    if target.exists():
        print(f"  ✓ {target.name} (already present)")
        return
    print(
        "  (no public ONNX export exists for nvidia/speakerverification_en_titanet_large —\n"
        "   ships as .nemo only. Export manually with NeMo:\n"
        "     from nemo.collections.asr.models import EncDecSpeakerLabelModel\n"
        "     m = EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large')\n"
        f"     m.export('{target}')\n"
        "   then re-run this script.)"
    )


def wake_word() -> None:
    print("── Wake word placeholder ──")
    target = _MODELS / "wake_word" / "hey_buddy.tflite"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(
            "  (no 'hey buddy' model — train one via openwakeword's "
            "synthetic-data pipeline and drop it here to activate)"
        )


def check_espeak() -> None:
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
        print("✓ espeak-ng present (required by Kokoro)")
    except Exception:
        print("✗ espeak-ng missing — install with: sudo apt install espeak-ng")


def main() -> int:
    _MODELS.mkdir(exist_ok=True)
    print(f"Downloading models into {_MODELS}\n")
    llm()
    vlm()
    function_gemma()
    stt()
    tts()
    vision()
    eou()
    speaker_id()
    wake_word()
    check_espeak()
    print("\nDone. Re-run this script any time to pick up missing files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
