# Companion

A fully-offline multimodal AI companion robot for the Jetson Orin Nano 8 GB.

It hears you, sees you, and animates a face on a small touchscreen that
the ESP32 module renders locally. 100 % of inference happens on-device —
no cloud, no API keys.

## Feature Status

> **Current runtime profile: Push-to-Talk chatbot + face tracking.**
> The codebase includes many additional features that are fully implemented
> but disabled on the 8 GB Orin to conserve VRAM. See `config.yaml` for
> the "Disabled Features" section at the bottom.

### ✅ Active (running in production)

| Feature | What it does |
|---|---|
| **LLM chat** | Llama 3.2 1B via llama-cpp-python, CUDA, streaming, KV-cache prefill |
| **STT (Parakeet)** | NVIDIA Parakeet 0.6B, streaming partial transcripts |
| **TTS (Piper)** | Lightweight, fast TTS. Kokoro also available via config swap |
| **Push-to-Talk** | Spacebar-driven turns; VAD detects speech-end |
| **Barge-in detection** | Adaptive noise floor + envelope AEC-lite + sustained-speech gating |
| **Vision + face detection** | CSI camera → YOLO26n-pose face detection at ~30 fps |
| **Face tracking** | Proportional 2-DOF head control via ST3215 servos |
| **ESP32 face display** | Animated face on Diymore 2.8" touchscreen over serial |
| **Affect-tagged expressions** | LLM `[affect: X]` tags fire 1.2 s expression overlays on the face |
| **Lip sync (envelope)** | Audio amplitude → viseme events → mouth animation |
| **Touchscreen UI** | Mute / Stop / Sleep / Volume / Restart tiles |
| **ReSpeaker DOA** | Direction-of-arrival + LED ring control |
| **BehaviorEngine** | 20 Hz motor + face-display coordinator |
| **Health watchdog** | 1 Hz checks: mic starvation, frozen camera, motor over-temp |
| **Telemetry** | Per-turn JSONL traces with phase timestamps |
| **Readiness gate** | Audits model files + serial ports at startup |
| **Layered config** | `config.yaml` ← `config.local.yaml` ← `COMPANION_*` env vars |

### 🔶 Disabled (fully implemented, off for 8 GB Orin constraints)

These features can be enabled by setting `enabled: true` in `config.yaml`.
Models are downloaded by `scripts/download_models.py`.

| Feature | Config toggle | Why it's off |
|---|---|---|
| **VLM / scene captioning** | `vlm.enabled` | Moondream needs ~3 GB VRAM — doesn't fit alongside the chat LLM |
| **Persistent memory** | `memory.enabled` | Mem0+Chroma add background load; needs `pip install mem0ai chromadb` |

## Hardware

| Device | Model |
|---|---|
| Compute | Jetson Orin Nano 8 GB, JetPack 6.x |
| Mic | Seeed ReSpeaker 4-Mic USB Array (`2886:0018`) |
| Speaker | Any USB audio device |
| Camera | CSI module at `cam0` (IMX219 or similar) |
| Touchscreen | Diymore ESP32 2.8" 240×320 (ILI9341 + XPT2046 + CH340) |

## Quick start

```bash
git clone <repo> companion && cd companion

# 1. System + CUDA + CH341 driver + venv + pip deps (≈20–30 min first time)
bash scripts/setup.sh

source companion_env/bin/activate

# 2. Download every model (LLM, STT, TTS, vision)
python3 scripts/download_models.py

# 3. Flash the ESP32 face firmware (with the screen plugged in)
bash scripts/flash_firmware.sh

# 4. (optional) Preflight — verifies every model path in config.yaml exists
python3 scripts/preflight.py

# 5. Run
python3 main.py
```

**Current mode is Push-to-Talk.** Hold spacebar to speak; release to
let the robot reply. To switch to continuous mode, change
`conversation.mode: continuous` in config.yaml. Tap the touchscreen to
mute / stop / sleep.

## Everyday commands

```bash
python3 -m tests.cli env                 # sanity check
python3 -m tests.cli audio               # live mic / DOA / VAD in the terminal
python3 -m tests.cli stt                 # 5 s record + transcribe
python3 -m tests.cli llm "hello"         # one-shot LLM
python3 -m tests.cli tts "hi there"      # synthesise + play
python3 -m tests.cli vision --seconds 10 # emotion pipeline benchmark
python3 -m tests.cli face happy          # drive the face to a preset
python3 -m tests.cli speaker enrol --name Yogee
python3 -m tests.cli mem search "interview"
python3 -m tests.cli tools "set a timer for 5 minutes"
python3 -m tests.cli all                 # run every subsystem sanity check

python3 -m tests.debug_gui               # tabbed debug window
python3 scripts/face_track_demo.py --sim  # standalone face-tracking demo
```

## Swapping models

Every model swap is one line in [config.yaml](config.yaml):

```yaml
llm:
  model: gemma-4-e2b      # or gemma-4-e4b
stt:
  backend: parakeet       # or whisper
tts:
  engine: kokoro          # or piper
  voice: af_heart         # af_bella, af_sarah, af_nicole, af_sky, …
display:
  backend: pygame         # or esp32_serial
```

Restart the app; that's it.

## Project layout

```
robotic-companion/
├── companion/              # production source code
│   ├── core/               config, event_bus, errors, gpu_arbiter, health,
│   │                       onnx_runtime, readiness, telemetry, logging
│   ├── audio/              io, vad, stt (Parakeet+Whisper), tts (Piper+Kokoro),
│   │                       respeaker, barge_in
│   ├── vision/             camera, face_detector, emotion_classifier,
│   │                       pipeline, scene_watcher, face_tracker
│   ├── llm/                engine (llama-cpp), prompt, memory, router
│   ├── tools/              registry + timer, volume, remind_me, stopwatch, time
│   ├── behavior/           engine (20 Hz motor + face-display tick) + tracking
│   ├── conversation/       manager, coordinator, states, turn
│   ├── display/            renderer, face-state, lip-sync, pygame & esp32_serial
│   ├── motor/              bus, kinematics, controller, calibration UI
│   └── ui/                 theme, shared widgets, main_window
├── tests/                  cli.py (terminal) + debug_gui.py (tabbed GUI)
├── scripts/                setup.sh, download_models.py, flash_firmware.sh,
│                           verify.py, preflight.py, face_track_demo.py
├── docs/                   technical report (HTML+PDF), executive summary, assets
├── deploy/                 systemd service + udev rules
├── firmware/companion_face/ ESP32 Arduino/PlatformIO face firmware
├── models/                 (downloaded by scripts/download_models.py)
│   ├── llm/                Llama 3.2 1B GGUF
│   ├── stt/                Parakeet TDT 0.6B ONNX
│   ├── tts/                Piper voice ONNX
│   ├── vad/                Silero VAD ONNX
│   ├── vision/             YOLO + HSEmotion + YuNet ONNX
│   └── vlm/                Moondream-2 GGUF (disabled, for testing)
├── data/chroma/            Mem0 vector DB (runtime, gitignored)
├── logs/                   JSONL per day (runtime, gitignored)
├── config.yaml             all knobs — active features top, disabled features bottom
├── setup_and_test.ipynb    interactive setup + sanity checks
├── main.py                 production entry point
└── README.md
```

## Config reference

Every subsystem reads its own dataclass section of [config.yaml](config.yaml).
See [companion/core/config.py](companion/core/config.py) for the full schema
— field defaults live there.

`config.yaml` is organized into two sections:
1. **Active Features** — everything running in production
2. **Disabled Features** — fully implemented but off for 8 GB Orin constraints

## Firmware

The touchscreen renders the face locally so the serial link only carries
small state commands (≈30 Hz). Protocol + pinout:
[firmware/companion_face/README.md](firmware/companion_face/README.md).

Flash with `bash scripts/flash_firmware.sh`.

## Architecture (one conversation turn)

```
Mic ──▶ VAD ──▶ (on speech end) ──▶ STT ───────▶ EOU ─▶ Router ─┐
                                                                  │
      ┌─── chat ◀──── Memory + Emotion + Scene hint injection ◀──┤
      │                                                           │
      ├─── VQA  ◀──── LLM multimodal (current frame, question) ◀─┤
      │                                                           │
      └─── tool ◀──── FunctionGemma(user turn) ──▶ Tool.invoke ◀──┘
      │
      ▼                         (tokens stream as they arrive)
   LLM ──tokens──▶ pysbd sentence splitter ──▶ TTS ──▶ Speaker
                                                ▼
                                        envelope → visemes
                                                ▼
                                     Display (face mouth animates)
                                                ▼
                                              ESP32 screen / HDMI
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/dev/ttyUSB0` missing | Re-run `scripts/setup.sh` — rebuilds WCH CH341 driver. |
| Kokoro fails on startup | Check `models/kokoro/` and `pip install kokoro-onnx`. The loud fallback falls back to Piper. |
| OOM on Gemma 4 E4B | `llm.model: gemma-4-e2b` in config.yaml, or boot Jetson into multi-user target. |
| No face on ESP32 | `display.backend: pygame` to confirm face logic on HDMI; then check `scripts/flash_firmware.sh` output. |
| Interruption ignored | Check `conversation.allow_interruption: true`. |
| ReSpeaker silent | Unplug / replug USB; ensure udev rule at `/etc/udev/rules.d/60-respeaker.rules`. |

## Credits

Built on top of llama.cpp, kokoro-onnx, Piper, Silero VAD, openWakeWord,
HSEmotion, NVIDIA Parakeet, NeMo TitaNet, LiveKit's EOU, Mem0, Chroma,
and TFT_eSPI.
