#!/usr/bin/env python3
"""Verify GPU / CUDA support for all components."""

import os, sys, glob

print("=" * 50)
print("  GPU / CUDA Verification")
print("=" * 50)

ok = 0
total = 3

# 1. PyTorch
print("\n1. PyTorch:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
        ok += 1
    else:
        print("   CPU only (no CUDA)")
        print("   Fix: pip install torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126")
except ImportError:
    print("   Not installed")

# 2. STT backend (openai-whisper with CUDA or faster-whisper)
print("\n2. STT (Speech-to-Text):")
stt_cuda = False
try:
    import ctranslate2
    try:
        types = ctranslate2.get_supported_compute_types("cuda")
        print(f"   faster-whisper: v{ctranslate2.__version__} — CUDA types: {types}")
        stt_cuda = True
    except Exception:
        print(f"   faster-whisper: v{ctranslate2.__version__} — CPU only")
except ImportError:
    print("   faster-whisper: not installed")

try:
    import whisper
    import torch
    if torch.cuda.is_available():
        print(f"   openai-whisper: CUDA via PyTorch — WILL USE GPU")
        stt_cuda = True
    else:
        print(f"   openai-whisper: installed, but PyTorch has no CUDA")
except ImportError:
    if not stt_cuda:
        print("   openai-whisper: not installed (pip install openai-whisper)")

if stt_cuda:
    ok += 1
else:
    print("   => STT will run on CPU (still works, just slower)")

# 3. llama-cpp-python (LLM)
print("\n3. llama-cpp-python (LLM):")
try:
    import llama_cpp

    pkg_dir = os.path.dirname(llama_cpp.__file__)
    parent = os.path.dirname(pkg_dir)
    cuda_libs = set()
    for d in [pkg_dir, parent]:
        cuda_libs.update(glob.glob(os.path.join(d, "**", "*ggml-cuda*"), recursive=True))
        cuda_libs.update(glob.glob(os.path.join(d + ".libs", "**", "*ggml-cuda*"), recursive=True))

    cuda_sos = sorted(set(os.path.realpath(f) for f in cuda_libs if f.endswith(".so")))
    print(f"   v{llama_cpp.__version__}")
    if cuda_sos:
        for f in cuda_sos:
            sz = os.path.getsize(f) // 1024
            print(f"   CUDA backend: {os.path.basename(f)} ({sz}KB)")
        ok += 1
    else:
        print("   NO CUDA backend (libggml-cuda.so not found)")
        print("   Rebuild: bash scripts/build_cuda.sh")
except ImportError:
    print("   Not installed")

print(f"\n{'=' * 50}")
print(f"  Result: {ok}/{total} components have GPU support")
if ok >= 2:
    print("  LLM + STT on GPU — great performance!")
elif ok >= 1:
    print("  Partial GPU. Run: bash scripts/build_cuda.sh")
elif ok == 0:
    print("  No GPU. Run: bash scripts/build_cuda.sh")
print("=" * 50)
