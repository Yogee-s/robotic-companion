#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Flash the ESP32 face firmware.
#
#  Auto-detects the screen's serial port. If it can't find one, it walks
#  you through an unplug/replug to identify the exact /dev/tty* device
#  that's your screen — no need to hardcode anything.
#
#    bash scripts/flash_firmware.sh           # build + upload
#    bash scripts/flash_firmware.sh --monitor # also open serial monitor
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/../firmware/companion_face"

# ── 1. PlatformIO available? ──────────────────────────────────────────
if ! command -v pio >/dev/null 2>&1; then
    echo "Installing PlatformIO..."
    pip install --user platformio >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── 2. Try the usual suspects first ───────────────────────────────────
probe_port() {
    for name in ttyCH341USB0 ttyUSB0 ttyACM0 ttyUSB1 ttyCH341USB1; do
        if [ -e "/dev/$name" ]; then
            echo "/dev/$name"
            return 0
        fi
    done
    return 1
}

PORT=""
if PORT=$(probe_port); then
    echo "Detected serial port automatically: $PORT"
else
    echo ""
    echo "No common serial port found. Let's figure out which one is the screen."
    echo ""
    echo "  [1/3] Make sure the screen IS currently plugged into the Jetson via USB,"
    read -r -p "        then press ENTER..." _

    before=$(mktemp)
    ls /dev/tty* 2>/dev/null | sort >"$before"

    echo ""
    echo "  [2/3] Now UNPLUG the screen's USB cable, wait 2 seconds,"
    read -r -p "        then press ENTER..." _
    sleep 1

    after=$(mktemp)
    ls /dev/tty* 2>/dev/null | sort >"$after"

    # Find the device that disappeared.
    removed=$(comm -23 "$before" "$after" | head -1 || true)
    rm -f "$before" "$after"

    if [ -z "$removed" ]; then
        echo ""
        echo "✗ No serial device disappeared when you unplugged. That means:"
        echo "   - the screen wasn't plugged in before, OR"
        echo "   - the port isn't enumerating (bad cable / bad USB port)."
        echo ""
        echo "  Check 'sudo dmesg | tail' for clues, or try a different cable."
        exit 1
    fi

    echo "  Identified: $removed"
    echo ""
    echo "  [3/3] PLUG THE SCREEN BACK IN,"
    read -r -p "        then press ENTER..." _
    sleep 2

    if [ ! -e "$removed" ]; then
        echo "✗ $removed didn't come back after replug. Try a different USB port."
        exit 1
    fi
    PORT="$removed"
    echo "  Using $PORT"
fi

# ── 3. Build + upload with the detected port ──────────────────────────
export PLATFORMIO_UPLOAD_PORT="$PORT"
export PLATFORMIO_MONITOR_PORT="$PORT"

echo ""
echo "==  Building  =="
pio run

echo ""
echo "==  Uploading via $PORT  =="
pio run -t upload

if [ "${1-}" = "--monitor" ]; then
    echo ""
    echo "==  Serial monitor on $PORT (Ctrl+C to exit)  =="
    pio device monitor --port "$PORT" --baud 115200
fi

echo ""
echo "Done. Screen should now be running the latest firmware."
