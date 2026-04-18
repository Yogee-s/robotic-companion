# Companion Face Firmware (ESP32 / Diymore 2.8")

Custom PlatformIO firmware for the robot's touchscreen face.

Renders the face locally and accepts state commands over USB-serial from
the Jetson at 115200 baud. See `../../companion/display/backends/esp32_serial.py`
for the Jetson-side driver.

## Build & flash

```bash
pip install platformio
bash ../../scripts/flash_firmware.sh
```

Or directly:

```bash
pio run
pio run -t upload
pio device monitor
```

## Protocol

### Jetson → ESP32 (line-delimited ASCII)

| Command | Example | Meaning |
|---|---|---|
| `FACE v=+0.72 a=+0.30 talk=1 listen=0 think=0 sleep=0 gaze=-12 privacy=0` | streamed at ~30 Hz | face pose update |
| `VISEME ahh` | `VISEME rest` | mouth shape during TTS |
| `SCENE face \| quickgrid \| morelist` | `SCENE quickgrid` | force a UI scene |
| `PRIVACY 0 \| 1` | `PRIVACY 1` | draw the "blindfolded" privacy band |

### ESP32 → Jetson

| Event | Example |
|---|---|
| Button tap | `BTN mute_mic`, `BTN stop_talking`, `BTN timer`, ... |
| Raw touch (debug) | `TOUCH 120 185` |
| Boot | `BOOT companion_face` |

## Wiring (Diymore ESP32 2.8" CYD-family)

| Signal | ESP32 pin |
|---|---|
| TFT_MOSI | 13 |
| TFT_MISO | 12 |
| TFT_SCK  | 14 |
| TFT_CS   | 15 |
| TFT_DC   | 2  |
| TFT_BL   | 21 |
| TOUCH_CS | 33 |
| TOUCH_IRQ| 36 |

Pinout is set in `platformio.ini`. Adjust `board_build.f_flash` or pin
defines if the specific variant of your Diymore board uses different SPI
mappings — CYD clones vary.
