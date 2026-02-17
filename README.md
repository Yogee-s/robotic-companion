# AI Companion — Jetson Orin Nano

A fully offline voice-activated AI companion running on Jetson Orin Nano 8GB.

Speak naturally and get fast, spoken responses — all processed locally on the GPU.

## How It Works

```
Microphone → VAD → STT → LLM → TTS → Speaker
              (Silero) (Whisper) (Llama)  (Piper)
```

Sentences are synthesized and played **while the LLM is still generating**,
so you hear the response as soon as the first sentence is ready.

## Project Structure

```
robotic-companion/
├── companion/              # Core modules
│   ├── audio_io.py         #   Mic capture & speaker playback
│   ├── vad.py              #   Voice Activity Detection (Silero ONNX)
│   ├── stt.py              #   Speech-to-Text (faster-whisper / openai-whisper)
│   ├── llm.py              #   LLM inference (llama-cpp-python, CUDA)
│   ├── tts.py              #   Text-to-Speech (Piper, cached model)
│   ├── respeaker.py        #   ReSpeaker USB mic array (DOA + LEDs)
│   ├── conversation.py     #   Streaming pipeline orchestrator
│   └── gui.py              #   PyQt5 GUI
├── scripts/                # Setup & build scripts
│   ├── pre_setup.sh        #   One-time system packages + venv + pip deps
│   ├── build_cuda.sh       #   Build CUDA packages (torch, llama-cpp, etc.)
│   ├── verify_gpu.py       #   Check GPU support for all components
│   └── download_models.py  #   Download LLM, TTS, STT models
├── tests/                  # Test scripts
│   ├── test_pipeline.py    #   End-to-end: mic → STT → LLM → TTS → speaker
│   └── test_respeaker_gui.py
├── models/                 # Downloaded models (gitignored)
├── config.yaml             # All tunable parameters
├── main.py                 # Application entry point
├── setup_and_test.ipynb    # Interactive setup notebook
└── requirements.txt        # Python dependencies (with install instructions)
```

## Quick Start (Fresh Install)

### 1. System packages + venv + Python deps (once)
```bash
cd ~/Desktop/robotic-companion
bash scripts/pre_setup.sh
```
This installs system packages (CUDA, espeak-ng, portaudio, etc.), creates the
`companion_env` Python venv, installs all pip dependencies, and configures ReSpeaker.

### 2. Build CUDA packages (~20-30 min, once)
```bash
source companion_env/bin/activate
bash scripts/build_cuda.sh
```
This builds/installs:
- **torch 2.8.0** (CUDA, from NVIDIA Jetson AI Lab)
- **llama-cpp-python** (built from source with CUDA)
- **ctranslate2** (for faster-whisper)
- **openai-whisper** (PyTorch CUDA STT backend)
- torchaudio shim + dependency fixes

### 3. Download models
```bash
python3 scripts/download_models.py
```

### 4. Verify GPU (optional)
```bash
python3 scripts/verify_gpu.py
```

### 5. Run
```bash
python3 main.py
```
Press **SPACE** to talk (push-to-talk mode). Toggle continuous mode in the GUI.

## Rebuilding the Environment

If you need to start fresh:
```bash
cd ~/Desktop/robotic-companion
rm -rf companion_env
bash scripts/pre_setup.sh
source companion_env/bin/activate
bash scripts/build_cuda.sh
python3 scripts/download_models.py
python3 main.py
```

## Hardware

- **Jetson Orin Nano 8GB** (JetPack 6.2 / L4T R36.5)
- **ReSpeaker USB Mic Array v3.1** (DOA + LEDs)
- **USB Speaker**

## Stack

| Component | Technology | Runs on |
|-----------|-----------|---------|
| LLM | llama-cpp-python — Llama 3.2 3B Q4_K_M | **GPU** |
| STT | openai-whisper (CUDA) / faster-whisper (CPU fallback) | GPU/CPU |
| TTS | Piper — hfc_female-medium | CPU |
| VAD | Silero VAD v5 (ONNX) | CPU |
| DOA | ReSpeaker USB HID (XVF-3800) | USB |
| GUI | PyQt5 | — |

## Key Optimizations

- **Streaming pipeline** — TTS starts as soon as the first sentence leaves the LLM
- **Cached TTS model** — Piper voice loaded once, reused for every utterance
- **Direct audio piping** — PCM streamed to aplay via stdin (no temp files)
- **GPU offload** — LLM runs on CUDA (~10-15 tok/s vs ~2 tok/s on CPU)
- **Push-to-talk + continuous mode** — spacebar PTT by default, toggle in GUI
- **Singlish toggle** — Switch to Singlish mode via GUI
- **STT warmup** — dummy transcription at startup eliminates cold-start lag
- **Time-based sentence flushing** — speaks partial output if LLM is slow
- **100% offline** — no internet or API calls needed after setup
