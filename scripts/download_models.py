#!/usr/bin/env python3
"""Download all models: LLM, TTS voice, and Whisper STT."""

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
    print(f"   Downloading...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", llm_path, url], check=True)
    print(f"   Done ({os.path.getsize(llm_path) // (1024*1024)} MB)")

# ── 2. Piper TTS voices ──
print("\n2. Piper TTS Voices")
piper_dir = os.path.join(model_dir, "piper")
os.makedirs(piper_dir, exist_ok=True)

voices = [
    ("en_US-hfc_female-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium"),
    ("en_US-lessac-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"),
    ("en_US-amy-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"),
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

# ── 3. Whisper STT model ──
print("\n3. Whisper STT (base.en)")
try:
    from faster_whisper import WhisperModel
    print("   Caching model (first load downloads it)...")
    m = WhisperModel("base.en", device="cpu", compute_type="int8")
    del m
    print("   Cached.")
except Exception as e:
    print(f"   Error: {e}")
    print("   Run: pip install faster-whisper")

# ── 4. espeak-ng check ──
print("\n4. espeak-ng (needed for TTS phonemizer)")
ret = os.system("dpkg -s espeak-ng >/dev/null 2>&1")
if ret == 0:
    print("   Installed")
else:
    print("   NOT installed — run: sudo apt install espeak-ng libespeak-ng-dev")

print(f"\n{'=' * 50}")
print("  All models ready.")
print("=" * 50)
