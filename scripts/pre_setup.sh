#!/usr/bin/env bash
# =============================================================================
# Pre-Setup: Run ONCE from a terminal before using the notebook.
#   cd ~/Desktop/robotic-companion && bash scripts/pre_setup.sh
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  AI Companion — Pre-Setup"
echo "============================================"

# ── 1. System packages ──
echo ""
echo "=== Installing system packages (requires sudo) ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-dev python3-pip python3-venv \
    portaudio19-dev libasound2-dev \
    python3-pyqt5 \
    libusb-1.0-0-dev \
    ffmpeg \
    cmake build-essential \
    curl wget git \
    cuda-nvcc-12-6 \
    libcublas-dev-12-6 \
    cuda-cupti-12-6 \
    espeak-ng libespeak-ng-dev

# ── 2. Create Python venv ──
echo ""
VENV="$PROJECT_DIR/companion_env"
if [ -d "$VENV" ]; then
    echo "=== Python venv already exists at $VENV ==="
else
    echo "=== Creating Python venv ==="
    python3 -m venv "$VENV" --system-site-packages
    echo "Created: $VENV"
fi

# Upgrade pip and install Python dependencies
echo "=== Installing Python dependencies ==="
"$VENV/bin/pip" install --upgrade pip setuptools wheel -q
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q

# Install ipykernel so notebook can use this venv
"$VENV/bin/pip" install -q ipykernel
"$VENV/bin/python3" -m ipykernel install --user --name companion --display-name "Companion"
echo "Jupyter kernel 'Companion' registered."

# ── 3. ReSpeaker USB permissions ──
echo ""
echo "=== Setting up ReSpeaker USB permissions ==="
UDEV_FILE="/etc/udev/rules.d/99-respeaker.rules"
cat << 'EOF' | sudo tee "$UDEV_FILE" > /dev/null
# ReSpeaker 4 Mic Array v3.1 — USB HID permissions + audio rebind
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0666", GROUP="plugdev"
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", RUN+="/bin/sh -c 'echo $kernel > /sys/bus/usb/drivers/usb/unbind; sleep 1; echo $kernel > /sys/bus/usb/drivers/usb/bind'"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "ReSpeaker udev rules installed."

# ── 4. Activate ReSpeaker ALSA ──
echo ""
echo "=== Activating ReSpeaker ALSA audio ==="
RESPEAKER_PATH=""
for d in /sys/bus/usb/devices/*/idVendor; do
    dir=$(dirname "$d")
    vendor=$(cat "$d" 2>/dev/null)
    product=$(cat "$dir/idProduct" 2>/dev/null)
    if [ "$vendor" = "2886" ] && [ "$product" = "0018" ]; then
        RESPEAKER_PATH=$(basename "$dir")
        break
    fi
done

if [ -n "$RESPEAKER_PATH" ]; then
    echo "Found ReSpeaker at $RESPEAKER_PATH — rebinding..."
    echo "$RESPEAKER_PATH" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null 2>&1 || true
    sleep 2
    echo "$RESPEAKER_PATH" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null 2>&1 || true
    sleep 2
    if grep -q "ArrayUAC10\|ReSpeaker" /proc/asound/cards 2>/dev/null; then
        echo "ReSpeaker registered as ALSA sound card."
    else
        echo "ReSpeaker not yet visible. Try unplugging and replugging USB."
    fi
else
    echo "ReSpeaker USB device not found. Plug it in and re-run."
fi

echo ""
echo "============================================"
echo "  Pre-setup complete!"
echo ""
echo "  Next steps:"
echo "    1. source $VENV/bin/activate"
echo "    2. bash scripts/build_cuda.sh   (CUDA packages, ~20-30 min)"
echo "    3. python3 scripts/download_models.py"
echo "    4. python3 main.py"
echo "============================================"
