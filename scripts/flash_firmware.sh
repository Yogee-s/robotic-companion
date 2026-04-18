#!/usr/bin/env bash
# Flash the ESP32 face firmware.
#
# Uses PlatformIO. Make sure the screen is plugged in via USB-C and
# /dev/ttyUSB0 is present (run scripts/setup.sh first to install the
# CH341 kernel driver).
#
# Usage:
#   bash scripts/flash_firmware.sh           # build + upload
#   bash scripts/flash_firmware.sh --monitor # also open serial monitor

set -euo pipefail
cd "$(dirname "$0")/../firmware/companion_face"

if ! command -v pio >/dev/null 2>&1; then
    echo "PlatformIO not found. Installing via pip..."
    pip install --user platformio
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -e /dev/ttyUSB0 ]; then
    echo "!! /dev/ttyUSB0 missing. Plug in the screen and confirm 'lsusb' shows QinHeng CH340."
    echo "   If the device is present but /dev/ttyUSB0 isn't, re-run scripts/setup.sh"
    echo "   to build the CH341 kernel driver."
    exit 1
fi

echo "== Building =="
pio run

echo "== Uploading =="
pio run -t upload

if [ "${1-}" = "--monitor" ]; then
    echo "== Serial monitor (Ctrl+C to exit) =="
    pio device monitor
fi
