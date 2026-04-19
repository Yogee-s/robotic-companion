# Companion

A fully-offline multimodal AI companion robot for the Jetson Orin Nano 8 GB.

It hears you, sees you, remembers you, answers visually ("what is this?"),
runs tools (timers, volume, reminders), and animates a face on a small
touchscreen that the ESP32 module renders locally. 100 % of inference
happens on-device — no cloud, no API keys.

## Features

- **Natural continuous conversation** — no wake word. The robot is
  always listening, but a turn only triggers when a face is visible, the
  utterance was long enough (≥400 ms voiced), and DOA lines up with the
  face. TV across the room is ignored; you get the robot's attention by
  standing in front of it and speaking.
- **Fast reply** — streaming STT → LLM KV-cache prefill → short-opener
  prompt → streaming TTS → persistent aplay. Target end-of-speech to
  first audio: ≤ 800 ms.
- **Interrupt anywhere** — speak while the robot is *thinking* or
  *speaking*; the in-flight turn cancels within ~300 ms. False-positive
  barge-in detection (noise-floor + Silero VAD + envelope AEC-lite)
  rejects door slams and the robot's own echo.
- **Embodied tracking** — two ST3215 servos in a differential bevel
  gearbox drive a 2-DOF head that smoothly tracks your face. Gain drops
  during `THINKING` so the head holds still while the LLM is composing.
- **Emotion-aware** — YOLO26n-pose + HSEmotion 8-class classifier
  produce valence/arousal; every turn, the current emotion is injected
  into the LLM prompt.
- **Affect-tagged expressions** — LLM replies end with a terminal tag
  `[affect: happy | curious | confused | surprised | affectionate | sad]`
  that fires a 1.2 s ornament on the ESP32 face.
- **Visemes** — TTS PCM envelope drives the mouth during speech.
- **Unified multimodal model** — the same Gemma-4 that handles chat also
  answers visual questions ("what is this?") and captions the scene in
  the background. Moondream was consolidated away.
- **Scene awareness** — background captioning at 0.5 Hz, paused
  automatically during active turns via `GPUArbiter`.
- **Persistent memory** — Mem0 + Chroma, per-speaker.
- **Speaker ID** — TitaNet-L embeddings.
- **Tool calling** — FunctionGemma 270 M sidecar routes "set a timer"
  and friends to callables.
- **Touchscreen face** — Diymore 2.8" display over ESP32 serial; four
  working tiles (Mute mic · Stop · Sleep · More → Volume / Restart).
- **Observable** — every turn writes a JSONL trace with phase timestamps
  (`logs/traces_<date>.jsonl`).
- **Watchdog + graceful degradation** — `HealthMonitor` detects starved
  mic / frozen camera / over-temp motor; `Coordinator` announces via TTS
  and falls back where possible.
- **Readiness gate** — `python main.py` fails loudly on a missing model
  file rather than booting half-dead.
- **Layered config** — `config.yaml` ← `config.local.yaml` (gitignored)
  ← `COMPANION_<SECTION>_<KEY>` env vars. Per-device overrides without
  touching the checked-in defaults.

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

# 4. (optional) Preflight — verifies every model path in config.yaml exists
python3 scripts/preflight.py

# 5. Run
python3 main.py
```

**No button to press.** Walk into view of the camera and speak
naturally. The robot will engage when it sees a face and hears enough
voiced speech. To mute / stop / sleep, tap the touchscreen.

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
│   ├── core/          config + event_bus + events + errors + gpu_arbiter
│   │                  + health + onnx_runtime + readiness + telemetry + logging + proactive
│   ├── audio/         io + vad + stt (Parakeet+Whisper) + tts (Kokoro+Piper)
│   │                  + eou + speaker_id + respeaker + lip_sync + barge_in
│   ├── vision/        camera + face_detector + emotion_classifier
│   │                  + pipeline + scene_watcher (+ legacy vlm.py for the debug GUI only)
│   ├── llm/           engine (Gemma 4 multimodal) + prompt + memory + function_gemma + router
│   ├── tools/         registry + timer + volume + remind_me + stopwatch + time_weather
│   ├── behavior/      engine (20 Hz motor + face-display tick) + tracking (per-state gain)
│   ├── conversation/  manager (Turn lifecycle + engagement gates + streaming) + coordinator
│   │                  + states + turn
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
