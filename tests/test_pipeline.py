#!/usr/bin/env python3
"""
Quick pipeline test: speak into the mic, hear the AI respond.

Usage:
    python3 tests/test_pipeline.py
"""

import io
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

RECORD_SECONDS = 4
LLM_MODEL = "models/llama-3.2-3b-instruct-q4_k_m.gguf"

# --------------------------------------------------------------------------- #
#  Voice model: prefer lessac-medium, fall back to amy-medium
# --------------------------------------------------------------------------- #
VOICE_MODELS = [
    "models/piper/en_US-lessac-medium.onnx",
    "models/piper/en_US-amy-medium.onnx",
]
VOICE_MODEL = next((v for v in VOICE_MODELS if os.path.exists(v)), VOICE_MODELS[-1])


def find_alsa_input():
    try:
        cards = open("/proc/asound/cards").read()
        if "ArrayUAC10" in cards or "ReSpeaker" in cards:
            return "plughw:ArrayUAC10,0"
    except Exception:
        pass
    return None


def find_alsa_output():
    try:
        cards = open("/proc/asound/cards").read()
        if "Device" in cards:
            return "plughw:Device,0"
    except Exception:
        pass
    return None


def play_pcm(pcm_bytes, sample_rate):
    """Play raw PCM int16 through the speaker."""
    speaker = find_alsa_output()
    for dev in [speaker, None]:
        cmd = ["aplay", "-q", "-f", "S16_LE", "-r", str(sample_rate),
               "-c", "1", "-t", "raw"]
        if dev:
            cmd.extend(["-D", dev])
        try:
            proc = subprocess.run(cmd, input=pcm_bytes, capture_output=True, timeout=30)
            if proc.returncode == 0:
                return True
        except Exception:
            continue
    return False


def chunk_to_pcm(chunk):
    """Convert a chunk from piper's synthesize() generator to raw PCM int16 bytes."""
    # AudioChunk object (piper-tts >= 2.x)
    if hasattr(chunk, "audio_int16_bytes"):
        data = chunk.audio_int16_bytes
        if data and len(data) > 0:
            return data
    if isinstance(chunk, bytes) and len(chunk) > 0:
        return chunk
    if isinstance(chunk, np.ndarray) and len(chunk) > 0:
        if chunk.dtype.kind == "f":
            return (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        return chunk.astype(np.int16).tobytes()
    if isinstance(chunk, tuple) and len(chunk) > 0:
        return chunk_to_pcm(chunk[0])
    return None


def synthesize_to_pcm(voice, text):
    """
    Synthesize text to raw PCM bytes using PiperVoice.

    Tries every known piper API, then falls back to CLI.
    """
    sample_rate = 22050
    try:
        sample_rate = voice.config.sample_rate
    except Exception:
        pass

    # --- Method 1: synthesize() generator ---
    # In piper-tts, this yields raw int16 PCM bytes (one chunk per sentence)
    # or numpy float32 arrays depending on version.
    try:
        chunks = []
        for i, item in enumerate(voice.synthesize(text)):
            pcm = chunk_to_pcm(item)
            if pcm:
                chunks.append(pcm)
                if i == 0:
                    print(f"    synthesize() yielding: type={type(item).__name__}, "
                          f"chunk_size={len(pcm)} bytes")
        if chunks:
            pcm = b"".join(chunks)
            print(f"    OK via Python API: {len(pcm)} bytes ({len(chunks)} chunks)")
            return pcm, sample_rate
        else:
            print("    synthesize() yielded 0 chunks (phonemizer may need espeak-ng)")
    except Exception as e:
        print(f"    synthesize() failed: {e}")

    # --- Method 2: synthesize_stream_raw ---
    if hasattr(voice, "synthesize_stream_raw"):
        try:
            chunks = list(voice.synthesize_stream_raw(text))
            if chunks:
                pcm = b"".join(chunks)
                print(f"    OK via synthesize_stream_raw: {len(pcm)} bytes")
                return pcm, sample_rate
        except Exception as e:
            print(f"    synthesize_stream_raw failed: {e}")

    # --- Method 3: synthesize_wav ---
    if hasattr(voice, "synthesize_wav"):
        try:
            wav_bytes = voice.synthesize_wav(text)
            if wav_bytes and len(wav_bytes) > 44:
                buf = io.BytesIO(wav_bytes)
                with wave.open(buf, "rb") as rf:
                    sample_rate = rf.getframerate()
                    pcm = rf.readframes(rf.getnframes())
                if pcm:
                    print(f"    OK via synthesize_wav: {len(pcm)} bytes")
                    return pcm, sample_rate
        except Exception as e:
            print(f"    synthesize_wav failed: {e}")

    # --- Method 4: piper CLI fallback ---
    print("    Falling back to piper CLI...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "pathvalidate"],
            capture_output=True, timeout=30,
        )
        tmp = tempfile.mktemp(suffix=".wav")
        cmd = ["piper", "--model", VOICE_MODEL, "--output_file", tmp]
        proc = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            with wave.open(tmp, "rb") as rf:
                sample_rate = rf.getframerate()
                pcm = rf.readframes(rf.getnframes())
            os.unlink(tmp)
            if pcm:
                print(f"    OK via CLI: {len(pcm)} bytes")
                return pcm, sample_rate
        else:
            stderr = proc.stderr.strip().split("\n")[-1] if proc.stderr else "unknown"
            print(f"    CLI error: {stderr}")
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as e:
        print(f"    CLI failed: {e}")

    return None, sample_rate


def main():
    print("=" * 50)
    print("  AI Companion — Pipeline Test")
    print("  Speak into the mic, hear the AI respond.")
    print("=" * 50)

    # ── 0. GPU check ──
    print("\n  GPU status:")
    try:
        import torch
        print(f"    PyTorch CUDA: {'YES — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO (CPU)'}")
    except ImportError:
        print("    PyTorch: not installed")

    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        print(f"    CTranslate2 CUDA: {'YES' if types else 'NO'}")
    except Exception:
        print("    CTranslate2 CUDA: NO (will use CPU for STT)")

    # ── 1. Record ──
    alsa_in = find_alsa_input()
    tmp_wav = tempfile.mktemp(suffix=".wav")

    print(f"\n  Recording {RECORD_SECONDS}s — speak now!")
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1",
           "-d", str(RECORD_SECONDS), tmp_wav]
    if alsa_in:
        cmd.extend(["-D", alsa_in])

    subprocess.run(cmd, capture_output=True, timeout=RECORD_SECONDS + 5)

    with wave.open(tmp_wav, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    os.unlink(tmp_wav)

    audio = np.frombuffer(raw, dtype=np.int16)
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    print(f"  Recorded ({len(audio)} samples, RMS={rms:.0f})")

    if rms < 10:
        print("\n  Mic is silent — check your ReSpeaker connection.")
        return

    # ── 2. STT ──
    print("\n  Transcribing...")
    audio_f32 = audio.astype(np.float32) / 32768.0

    from faster_whisper import WhisperModel
    t0 = time.time()
    try:
        stt = WhisperModel("base.en", device="cuda", compute_type="int8")
        stt_device = "CUDA"
    except Exception:
        stt = WhisperModel("base.en", device="cpu", compute_type="int8")
        stt_device = "CPU"

    segs, _ = stt.transcribe(
        audio_f32, beam_size=1, language="en",
        vad_filter=True, without_timestamps=True,
    )
    text = " ".join(s.text for s in segs).strip()
    t_stt = time.time() - t0
    del stt

    if not text:
        print("  Nothing detected. Try speaking louder.")
        return

    print(f"  You said: \"{text}\" ({t_stt:.1f}s, {stt_device})")

    # ── 3. LLM ──
    print("\n  Thinking...")
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(
        model_path=LLM_MODEL, n_gpu_layers=-1,
        n_ctx=2048, n_batch=512, verbose=False,
    )
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content":
             "You are a friendly AI companion. Keep responses to 1-2 sentences."},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
    )
    reply = resp["choices"][0]["message"]["content"].strip()
    t_llm = time.time() - t0
    toks = resp.get("usage", {}).get("completion_tokens", "?")
    tok_s = f"{int(toks) / t_llm:.1f}" if isinstance(toks, int) else "?"
    del llm

    if not reply:
        print("  LLM returned empty response.")
        return

    print(f"  AI says: \"{reply}\" ({t_llm:.1f}s, ~{toks} tokens, ~{tok_s} tok/s)")

    # ── 4. TTS → Speaker ──
    print("\n  Speaking through speaker...")
    from piper import PiperVoice

    voice = PiperVoice.load(VOICE_MODEL)
    t0 = time.time()
    pcm, sr = synthesize_to_pcm(voice, reply)
    t_tts = time.time() - t0

    if pcm is None or len(pcm) == 0:
        print("  TTS produced no audio.")
        print("  TIP: Run 'sudo apt install espeak-ng' and try again.")
        return

    duration = len(pcm) / 2 / sr
    print(f"  Synthesized {duration:.1f}s of audio ({t_tts:.2f}s)")

    ok = play_pcm(pcm, sr)
    if ok:
        print("  Done!")
    else:
        print("  Could not play audio — check speaker connection.")

    print(f"\n  Total: STT={t_stt:.1f}s  LLM={t_llm:.1f}s  TTS={t_tts:.1f}s")
    total = t_stt + t_llm + t_tts
    print(f"  Pipeline total: {total:.1f}s")
    if total > 10:
        print("  TIP: Rebuild llama-cpp-python and ctranslate2 with CUDA")
        print("       to bring this under 5 seconds. See setup_and_test.ipynb.")
    print()


if __name__ == "__main__":
    main()
