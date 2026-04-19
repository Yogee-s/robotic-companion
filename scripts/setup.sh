#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Companion — one-shot setup script for Jetson Orin Nano (JetPack 6.x)
#
#  Idempotent. Installs:
#    1. System packages  (CUDA dev, espeak-ng, portaudio, v4l-utils, PyQt5 deps)
#    2. Python venv + pip requirements
#    3. CUDA-specific wheels (torch, llama-cpp-python, ctranslate2) from the
#       Jetson AI Lab index
#    4. WCH CH341SER kernel driver  (so /dev/ttyUSB0 appears for the screen)
#    5. ReSpeaker USB udev rule
#
#  Usage:   bash scripts/setup.sh
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

log() { echo -e "\e[1;34m==>\e[0m $*"; }
warn() { echo -e "\e[1;33m!! \e[0m $*"; }
ok() { echo -e "\e[1;32m ✓ \e[0m $*"; }

# ── 1. System packages ─────────────────────────────────────────────────────
log "Installing system packages (sudo)..."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-pip python3-dev \
    build-essential cmake ninja-build git pkg-config \
    portaudio19-dev libportaudio2 alsa-utils \
    espeak-ng libespeak-ng-dev \
    v4l-utils libv4l-dev \
    libgl1 libglib2.0-0 \
    qtbase5-dev libqt5gui5 libqt5widgets5 libxkbcommon-x11-0 \
    cuda-nvcc-12-6 libcublas-dev-12-6 \
    linux-headers-$(uname -r) \
    usbutils

# ── 2. venv + pip requirements ─────────────────────────────────────────────
if [ ! -d "companion_env" ]; then
    log "Creating venv..."
    python3 -m venv companion_env
fi
# shellcheck disable=SC1091
source companion_env/bin/activate

log "Upgrading pip + wheel..."
pip install --upgrade pip wheel setuptools

log "Installing Python dependencies..."
pip install -r requirements.txt

# kokoro-onnx: metadata pins numpy>=2 (conflicts with Jetson torch's numpy<2),
# but runtime works on numpy 1.26.x. Install with --no-deps to skip the pin.
if ! python -c "import kokoro_onnx" 2>/dev/null; then
    log "Installing kokoro-onnx (no-deps, Jetson numpy workaround)..."
    pip install "kokoro-onnx>=0.4.0" --no-deps
else
    ok "kokoro-onnx already installed"
fi

# ── 3. CUDA-specific wheels from Jetson AI Lab ─────────────────────────────
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    log "Installing torch 2.8.0 with CUDA from $JETSON_INDEX ..."
    pip install torch==2.8.0 --index-url "$JETSON_INDEX" --no-cache-dir --no-deps
else
    ok "torch CUDA already installed"
fi

if ! python -c "import ctranslate2" 2>/dev/null; then
    log "Installing ctranslate2 from Jetson index..."
    pip install ctranslate2 --index-url "$JETSON_INDEX" --no-cache-dir || warn "ctranslate2 install failed"
fi

if ! python -c "from llama_cpp import Llama" 2>/dev/null; then
    log "Building llama-cpp-python with CUDA (≈20 min first time)..."
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" \
        FORCE_CMAKE=1 \
        pip install llama-cpp-python --no-cache-dir
else
    ok "llama-cpp-python already installed"
fi

# ── 4. WCH CH340/CH341 serial driver (for the ESP32 screen) ────────────────
# juliagoda's fork builds as module "ch34x" (not "ch341"). This rebuild
# step must be re-run after every kernel/JetPack upgrade — out-of-tree
# modules don't survive kernel updates.
if ! lsmod | grep -q ch34; then
    log "Building WCH CH34x kernel driver..."
    tmp="$(mktemp -d)"
    (
        cd "$tmp"
        git clone --depth=1 https://github.com/juliagoda/CH341SER.git CH341SER
        cd CH341SER
        make clean >/dev/null 2>&1 || true
        make
        sudo make install
    )
    sudo modprobe ch34x
    if ! grep -q ch34x /etc/modules-load.d/ch34x.conf 2>/dev/null; then
        echo ch34x | sudo tee /etc/modules-load.d/ch34x.conf >/dev/null
    fi
    # Clean up any stale ch341 entry from older setups
    sudo rm -f /etc/modules-load.d/ch341.conf
    ok "CH34x driver installed"
else
    ok "CH34x driver already loaded"
fi

# ── 5. ReSpeaker USB udev rule ─────────────────────────────────────────────
RESPEAKER_RULE="/etc/udev/rules.d/60-respeaker.rules"
if [ ! -f "$RESPEAKER_RULE" ]; then
    log "Installing ReSpeaker udev rule..."
    echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"' \
        | sudo tee "$RESPEAKER_RULE" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ok "ReSpeaker udev rule installed (unplug/replug USB if needed)"
else
    ok "ReSpeaker udev rule already present"
fi

ok "Setup complete. Next:"
echo "   source companion_env/bin/activate"
echo "   python3 scripts/download_models.py"
echo "   bash   scripts/flash_firmware.sh   # optional — flashes the ESP32 face"
echo "   python3 main.py"
