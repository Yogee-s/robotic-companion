#!/usr/bin/env bash
# ============================================================
# Build CUDA-accelerated packages for Jetson Orin Nano.
# Run once — takes ~20-30 min.
#
# Usage:
#   source companion_env/bin/activate
#   bash scripts/build_cuda.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/companion_env"

# Activate venv
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "Using venv: $VENV"
    echo "Python: $(which python3)"
else
    echo "ERROR: venv not found at $VENV"
    echo "Run first: bash scripts/pre_setup.sh"
    exit 1
fi

echo ""
echo "============================================"
echo "  GPU Package Builder (Jetson Orin Nano)"
echo "============================================"

# ── 1. Set CUDA environment ──
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87"
export FORCE_CMAKE=1

echo ""
echo "CUDA_HOME  = $CUDA_HOME"
echo "CUDACXX    = $CUDACXX"
echo "CMAKE_ARGS = $CMAKE_ARGS"

# Verify nvcc exists
if ! command -v nvcc &>/dev/null; then
    echo ""
    echo "ERROR: nvcc not found. Run:"
    echo "  sudo apt-get install -y cuda-nvcc-12-6 libcublas-dev-12-6 cuda-cupti-12-6"
    exit 1
fi

# Verify cublas dev headers exist
if [ ! -f /usr/local/cuda/lib64/libcublas.so ]; then
    echo ""
    echo "ERROR: libcublas not found. Run:"
    echo "  sudo apt-get install -y libcublas-dev-12-6"
    exit 1
fi

echo ""
echo "CUDA toolkit OK."

# Make sure pip is up to date (old pip causes build failures)
pip install --upgrade pip setuptools wheel -q

# ── 2. Install CUDA PyTorch from NVIDIA Jetson AI Lab ──
echo ""
echo "============================================"
echo "  Installing CUDA PyTorch 2.8.0 (Jetson)"
echo "============================================"
pip install "torch==2.8.0" \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    --no-cache-dir --no-deps 2>&1 | tail -5

# Pin it so other packages don't overwrite with CPU version
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  torch {torch.__version__} — CUDA OK')"

# ── 3. Install numpy <2 (required by torch 2.8 Jetson wheel) ──
pip install "numpy>=1.24,<2" -q

# ── 4. Build llama-cpp-python with CUDA ──
echo ""
echo "============================================"
echo "  Building llama-cpp-python with CUDA"
echo "  This takes 20-30 min. Be patient."
echo "============================================"
echo ""

pip uninstall -y llama-cpp-python 2>/dev/null || true
pip install llama-cpp-python --no-cache-dir --no-binary llama-cpp-python 2>&1 | \
    grep -E "cmake|CUDA|cuda|ggml|Building|Successfully|ERROR|error" || true

# Verify
echo ""
python3 "$SCRIPT_DIR/verify_gpu.py" 2>/dev/null | grep -A2 "llama-cpp" || true

# ── 5. Install CTranslate2 (pre-built wheel, CPU-only on aarch64) ──
echo ""
echo "============================================"
echo "  Installing CTranslate2 (for faster-whisper)"
echo "============================================"
pip install ctranslate2 --no-deps 2>&1 | tail -3

# ── 6. Install openai-whisper (PyTorch CUDA STT backend) ──
echo ""
echo "============================================"
echo "  Installing openai-whisper (CUDA STT)"
echo "============================================"
pip install openai-whisper --no-deps 2>&1 | tail -3
# Install whisper's deps WITHOUT letting it pull torch
pip install tiktoken more-itertools -q

# ── 7. Fix dependency issues ──
echo ""
echo "============================================"
echo "  Fixing dependency compatibility"
echo "============================================"

# numba + coverage fix
pip install "coverage>=7.0" "numba>=0.60" -q

# Remove torchaudio if it snuck in (incompatible with Jetson torch 2.8)
pip uninstall -y torchaudio 2>/dev/null || true

# Create torchaudio shim (silero-vad imports it but doesn't need native libs)
SITE_PKG="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SITE_PKG/torchaudio"
cat > "$SITE_PKG/torchaudio/__init__.py" << 'SHIM'
"""Minimal torchaudio shim for silero-vad (no native libs needed)."""
__version__ = "0.0.0"
def load(*a, **kw): raise NotImplementedError("torchaudio shim")
class transforms:
    class Resample:
        def __init__(self, *a, **kw): pass
        def __call__(self, x): return x
SHIM
echo "  torchaudio shim installed"

# Verify torch is still CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'torch CUDA broken!'; print(f'  torch {torch.__version__} CUDA OK')"

echo ""
echo "============================================"
echo "  Done! Run: python3 scripts/verify_gpu.py"
echo "============================================"
