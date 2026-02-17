#!/usr/bin/env python3
"""Download all models: LLM, Kokoro TTS, Piper TTS (fallback), and Whisper STT."""

import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

print("=" * 50)
print("  Model Downloader")
print("=" * 50)

# ── 1. LLM ──
print("\n1. LLM — Llama 3.2 3B Q4_K_M (~2 GB)")
model_dir = os.path.join(PROJECT, "models")
os.makedirs(model_dir, exist_ok=True)
llm_path = os.path.join(model_dir, "llama-3.2-3b-instruct-q4_k_m.gguf")

if os.path.exists(llm_path):
    print(f"   Already exists ({os.path.getsize(llm_path) // (1024*1024)} MB)")
else:
    url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    print("   Downloading...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", llm_path, url], check=True)
    print(f"   Done ({os.path.getsize(llm_path) // (1024*1024)} MB)")

# ── 2. Kokoro TTS (primary — natural voice) ──
print("\n2. Kokoro TTS — natural voice (~300 MB)")
kokoro_dir = os.path.join(model_dir, "kokoro")
os.makedirs(kokoro_dir, exist_ok=True)

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
kokoro_files = [
    ("kokoro-v1.0.fp16.onnx", f"{KOKORO_BASE}/kokoro-v1.0.fp16.onnx"),
    ("voices-v1.0.bin", f"{KOKORO_BASE}/voices-v1.0.bin"),
]

for name, url in kokoro_files:
    path = os.path.join(kokoro_dir, name)
    if os.path.exists(path):
        print(f"   {name}: already exists ({os.path.getsize(path) // (1024*1024)} MB)")
    else:
        print(f"   {name}: downloading...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", path, url], check=True)
        print(f"   {name}: done ({os.path.getsize(path) // (1024*1024)} MB)")

# ── 3. Piper TTS voices (fallback) ──
print("\n3. Piper TTS Voices (fallback)")
piper_dir = os.path.join(model_dir, "piper")
os.makedirs(piper_dir, exist_ok=True)

voices = [
    ("en_US-hfc_female-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium"),
]

for name, base_url in voices:
    onnx = os.path.join(piper_dir, f"{name}.onnx")
    if os.path.exists(onnx):
        print(f"   {name}: already exists")
    else:
        print(f"   {name}: downloading...")
        subprocess.run(["wget", "-q", "-O", onnx, f"{base_url}/{name}.onnx"], check=True)
        subprocess.run(["wget", "-q", "-O", f"{onnx}.json", f"{base_url}/{name}.onnx.json"], check=True)
        print(f"   {name}: done")

# ── 4. Whisper STT models ──
print("\n4. Whisper STT")
try:
    from faster_whisper import WhisperModel
    for wmodel in ["base.en", "small.en"]:
        print(f"   Caching {wmodel} (first load downloads it)...")
        m = WhisperModel(wmodel, device="cpu", compute_type="int8")
        del m
        print(f"   {wmodel}: cached.")
except Exception as e:
    print(f"   Error: {e}")
    print("   Run: pip install faster-whisper")

# ── 5. espeak-ng check ──
print("\n5. espeak-ng (needed for TTS phonemizer)")
ret = os.system("dpkg -s espeak-ng >/dev/null 2>&1")
if ret == 0:
    print("   Installed")
else:
    print("   NOT installed — run: sudo apt install espeak-ng libespeak-ng-dev")

print(f"\n{'=' * 50}")
print("  All models ready.")
print(f"{'=' * 50}")
