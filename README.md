# Companion

A fully-offline multimodal AI companion robot for the Jetson Orin Nano 8 GB.

It hears you, sees you, remembers you, answers visually ("what is this?"),
runs tools (timers, volume, reminders), and animates a face on a small
touchscreen that the ESP32 module renders locally. 100 % of inference
happens on-device — no cloud, no API keys.

## Features

- **Natural voice loop** — mic → VAD → STT → LLM → TTS → speaker with
  streaming: TTS starts on the first sentence while the LLM is still
  generating.
- **Interruption** — speak while the robot is replying; it stops within
  ~300 ms and listens.
- **Semantic end-of-turn** — a small transformer decides whether you're
  actually finished or just paused, so the robot stops cutting you off.
- **Emotion-aware** — the camera runs an 8-emotion classifier and a
  valence/arousal signal is injected into the LLM prompt (gated so it
  never spams context).
- **Scene awareness** — Moondream-2 captions the camera feed 1–2 times a
  second, and the caption is available to the LLM as context.
- **Visual Q&A** — ask "what am I holding?" / "can you read this?" and
  the VLM answers from the current frame.
- **Persistent memory** — Mem0 + local Chroma, per-speaker scope. Next
  time you talk, the robot can recall what you said yesterday.
- **Speaker ID** — TitaNet-L embeddings identify who is speaking so
  memory + tone can be personalised.
- **Wake word** — optional "Hey Buddy" idle-listening mode via
  openWakeWord (disabled by default).
- **Tool calling** — a 270 M FunctionGemma sidecar routes *"set a timer
  for 5 minutes"* and friends to real callable tools.
- **Touchscreen face** — custom ESP32 firmware on a Diymore 2.8"
  240×320 display renders the face locally; Jetson sends tiny state
  commands over serial at 30 Hz; tap the screen to reveal 4 big buttons
  (mute / stop / sleep / more). Pygame fallback runs on HDMI for
  development.
- **DOA-driven gaze** — the face's eyes glance toward whoever is
  speaking, using ReSpeaker beam-forming.
- **Lip-sync** — Rhubarb viseme timings drive the mouth during TTS
  playback.
- **Proactive mode (opt-in)** — the robot can greet familiar faces and
  check in when you seem sad. Rate-limited.
- **Privacy toggle** — cover the camera in software (face shows a
  "blindfold" band).

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

# 2. Download every model (LLM, VLM, STT, TTS, vision, EOU, speaker-ID)
python3 scripts/download_models.py

# 3. Flash the ESP32 face firmware (with the screen plugged in)
bash scripts/flash_firmware.sh

# 4. Run
python3 main.py
```

Press **SPACE** to talk.

## Everyday commands

```bash
python3 -m tests.cli env                 # sanity check
python3 -m tests.cli audio               # live mic / DOA / VAD in the terminal
python3 -m tests.cli stt                 # 5 s record + transcribe
python3 -m tests.cli llm "hello"         # one-shot LLM
python3 -m tests.cli vlm "what do you see?"
python3 -m tests.cli tts "hi there"      # synthesise + play
python3 -m tests.cli vision --seconds 10 # emotion pipeline benchmark
python3 -m tests.cli face happy          # drive the face to a preset
python3 -m tests.cli speaker enrol --name Yogee
python3 -m tests.cli mem search "interview"
python3 -m tests.cli tools "set a timer for 5 minutes"
python3 -m tests.cli all                 # run every subsystem sanity check

python3 -m tests.debug_gui               # one window, 5 tabs — Audio/LLM/TTS/Vision/Face
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
├── companion/
│   ├── core/          config.py + logging.py + events.py + proactive.py
│   ├── audio/         io + vad + stt (Parakeet+Whisper) + tts (Kokoro+Piper)
│   │                  + eou + wake_word + speaker_id + respeaker
│   ├── vision/        camera + face_detector (YuNet) + emotion_classifier
│   │                  + pipeline + vlm (Moondream) + scene_watcher
│   ├── llm/           engine (Gemma 4) + prompt + memory + function_gemma + router
│   ├── tools/         registry + timer + volume + remind_me + stopwatch + time_weather
│   ├── conversation/  manager — orchestrates the pipeline
│   ├── display/       renderer + face-state + lip-sync + pygame & esp32_serial backends
│   └── ui/            theme + shared widgets + main_window
├── tests/             cli.py (terminal) + debug_gui.py (tabbed)
├── scripts/           setup.sh + download_models.py + flash_firmware.sh + verify.py
├── firmware/companion_face/  ESP32 Arduino/PlatformIO project
├── models/            (downloaded)
├── data/chroma/       (Mem0 vector DB)
├── logs/              (JSONL per day)
├── config.yaml        all knobs
├── main.py
└── README.md
```

## Config reference

Every subsystem reads its own dataclass section of [config.yaml](config.yaml).
See [companion/core/config.py](companion/core/config.py) for the full schema
— field defaults live there.

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
      ├─── VQA  ◀──── Moondream(current frame, question) ◀────────┤
      │                                                           │
      └─── tool ◀──── FunctionGemma(user turn) ──▶ Tool.invoke ◀──┘
      │
      ▼                         (tokens stream as they arrive)
   LLM ──tokens──▶ pysbd sentence splitter ──▶ TTS ──▶ Speaker
                                                ▼
                                              Rhubarb → visemes
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
HSEmotion, YuNet, Moondream 2, NVIDIA Parakeet, NeMo TitaNet, LiveKit's
EOU, Mem0, Chroma, TFT_eSPI, and Rhubarb Lip Sync.
