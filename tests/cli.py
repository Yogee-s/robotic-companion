"""Terminal test runner — headless CLI that exercises individual subsystems.

Usage:
    python3 -m tests.cli env          # system + GPU sanity check
    python3 -m tests.cli audio        # 10 s mic capture + RMS/DOA/VAD dump
    python3 -m tests.cli stt          # record 5 s and transcribe
    python3 -m tests.cli llm "hi"     # one-shot LLM generation
    python3 -m tests.cli vlm "what?"  # caption current camera frame
    python3 -m tests.cli tts "hello"  # synthesise + play a sentence
    python3 -m tests.cli vision       # 10 s emotion pipeline benchmark
    python3 -m tests.cli mem search "job interview"
    python3 -m tests.cli speaker enrol --name Yogee
    python3 -m tests.cli tools "set a timer for 5 minutes"
    python3 -m tests.cli face happy   # drive the display face to a preset
    python3 -m tests.cli pipeline     # end-to-end single turn
    python3 -m tests.cli all          # run every subsystem sanity check

Designed to be runnable over SSH — no GUI required.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on sys.path when invoked as `python3 -m tests.cli`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from companion.core.config import load_config  # noqa: E402
from companion.core.logging import setup_logging  # noqa: E402


def _load_cfg():
    cfg = load_config(os.path.join(_ROOT, "config.yaml"))
    setup_logging(cfg.app.log_level)
    return cfg


def cmd_env(_args) -> int:
    cfg = _load_cfg()
    print("── Environment ─────────────────────────────")
    print(f"Project root : {cfg.project_root}")
    try:
        import torch

        print(f"torch        : {torch.__version__}  CUDA={torch.cuda.is_available()}")
    except ImportError:
        print("torch        : not installed")
    try:
        import onnxruntime as ort

        print(f"onnxruntime  : {ort.__version__}  providers={ort.get_available_providers()}")
    except ImportError:
        print("onnxruntime  : not installed")
    try:
        import llama_cpp

        print(f"llama-cpp    : {llama_cpp.__version__}")
    except (ImportError, AttributeError):
        print("llama-cpp    : not installed")

    print("── Devices ─────────────────────────────────")
    for name, check in (
        ("ttyUSB0 (screen)", lambda: os.path.exists(cfg.display.serial_port)),
        ("CSI camera cam0",  lambda: os.path.exists("/dev/video0")),
    ):
        print(f"  {name:<20} {'✓' if check() else '✗'}")
    return 0


def cmd_audio(args) -> int:
    cfg = _load_cfg()
    from companion.audio.io import AudioInput
    from companion.audio.respeaker import ReSpeakerArray
    from companion.audio.vad import VoiceActivityDetector
    import numpy as np

    rs = ReSpeakerArray({"vendor_id": cfg.respeaker.vendor_id, "product_id": cfg.respeaker.product_id})
    ai = AudioInput({
        "sample_rate": cfg.audio.sample_rate,
        "channels": cfg.audio.channels,
        "chunk_size": cfg.audio.chunk_size,
        "input_device_name": cfg.audio.input_device_name,
    })
    vad = VoiceActivityDetector({
        "threshold": cfg.vad.threshold,
        "silence_duration_ms": cfg.vad.silence_duration_ms,
        "min_speech_duration_ms": cfg.vad.min_speech_duration_ms,
    })
    ai.start()
    dur = float(args.seconds)
    t0 = time.time()
    last_doa = None
    last_print = 0.0
    try:
        while time.time() - t0 < dur:
            chunk = ai.read(timeout=0.5)
            if chunk is None:
                continue
            vad.process_chunk(chunk)
            rms = float(np.sqrt(np.mean(chunk**2)))
            doa, _voice = rs.get_doa(), False
            now = time.time()
            if now - last_print > 0.2:
                last_print = now
                vad_prob = float(getattr(vad, "last_prob", 0.0))
                print(f"  t={now - t0:5.2f}s  RMS={rms:.3f}  DOA={doa:4.0f}°  VAD={vad_prob:.2f}")
            last_doa = doa
    finally:
        ai.stop()
        rs.stop()
    print(f"Done. last DOA: {last_doa}")
    return 0


def cmd_stt(_args) -> int:
    cfg = _load_cfg()
    from companion.audio.io import AudioInput
    from companion.audio.stt import SpeechToText
    import numpy as np

    stt = SpeechToText(cfg.stt, project_root=cfg.project_root)
    if not stt.is_loaded:
        print("STT backend failed to load"); return 2
    stt.warmup()

    ai = AudioInput({
        "sample_rate": cfg.audio.sample_rate,
        "channels": cfg.audio.channels,
        "chunk_size": cfg.audio.chunk_size,
        "input_device_name": cfg.audio.input_device_name,
    })
    ai.start()
    print("Recording for 5 seconds... speak now.")
    t0 = time.time()
    chunks: list[np.ndarray] = []
    try:
        while time.time() - t0 < 5.0:
            c = ai.read(timeout=0.5)
            if c is not None:
                chunks.append(c)
    finally:
        ai.stop()
    if not chunks:
        print("No audio captured"); return 2
    audio = np.concatenate(chunks)
    t_start = time.time()
    text = stt.transcribe(audio)
    print(f"Backend  : {stt.backend}")
    print(f"Latency  : {time.time() - t_start:.2f} s")
    print(f"Text     : {text!r}")
    return 0


def cmd_llm(args) -> int:
    cfg = _load_cfg()
    from companion.llm.engine import LLMEngine

    llm = LLMEngine(cfg.llm, model_path=cfg.llm_model_path())
    llm.load()
    prompt = " ".join(args.prompt) or "Say hi in one short sentence."
    t0 = time.time()
    out = llm.generate(user_message=prompt, history=[], system_prompt=cfg.llm.system_prompt)
    dt = time.time() - t0
    toks = len(out.split())
    print(f"[{dt:.2f}s, ~{toks / dt:.1f} tok/s]  {out}")
    return 0


def cmd_vlm(args) -> int:
    cfg = _load_cfg()
    from companion.vision.camera import CSICamera
    from companion.vision.vlm import MoondreamVLM

    cam = CSICamera(
        sensor_id=cfg.vision.sensor_id,
        width=cfg.vision.width,
        height=cfg.vision.height,
        fps=cfg.vision.fps,
        flip_method=cfg.vision.flip_method,
        use_csi=cfg.vision.use_csi,
    )
    vlm = MoondreamVLM(
        cfg.abspath(cfg.vlm.model_path), cfg.abspath(cfg.vlm.mmproj_path),
        enabled=cfg.vlm.enabled, max_tokens=cfg.vlm.max_tokens
    )
    if not vlm.available:
        print("VLM unavailable."); return 2
    frame = None
    for _ in range(40):
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.1)
    cam.close()
    if frame is None:
        print("No frame captured."); return 2
    question = " ".join(args.question) or "What do you see?"
    t0 = time.time()
    answer = vlm.answer(frame, question) if question != "caption" else vlm.caption(frame)
    print(f"[{time.time() - t0:.2f}s] {answer}")
    return 0


def cmd_tts(args) -> int:
    cfg = _load_cfg()
    from companion.audio.io import AudioOutput
    from companion.audio.tts import TextToSpeech

    tts = TextToSpeech(cfg.tts, project_root=cfg.project_root)
    out = AudioOutput({"output_sample_rate": tts.output_sample_rate})
    sentence = " ".join(args.text) or "Hello, I am your companion."
    pcm = tts.synthesize(sentence)
    if pcm is None:
        print("Synthesis failed."); return 2
    out.play_pcm(pcm, tts.output_sample_rate)
    print(f"Spoke {len(pcm) / 2 / tts.output_sample_rate:.1f} s")
    return 0


def cmd_vision(args) -> int:
    cfg = _load_cfg()
    from companion.vision import EmotionPipeline

    pipe = EmotionPipeline({
        "sensor_id": cfg.vision.sensor_id, "width": cfg.vision.width,
        "height": cfg.vision.height, "fps": cfg.vision.fps,
        "flip_method": cfg.vision.flip_method, "use_csi": cfg.vision.use_csi,
        "face_model_path": cfg.abspath(cfg.vision.face_model_path),
        "emotion_model_path": cfg.abspath(cfg.vision.emotion_model_path),
        "face_score_threshold": cfg.vision.face_score_threshold,
        "smoothing": cfg.vision.smoothing,
        "staleness_fade_seconds": cfg.vision.staleness_fade_seconds,
    })
    pipe.start()
    seen = False
    try:
        t0 = time.time()
        last_label = None
        while time.time() - t0 < args.seconds:
            s = pipe.get_state()
            if s.has_face:
                seen = True
                if s.label != last_label:
                    print(f"  {s.label:<10} conf={s.confidence*100:5.1f}% v={s.valence:+.2f} a={s.arousal:+.2f} {s.fps:5.1f} fps")
                    last_label = s.label
            time.sleep(0.1)
    finally:
        pipe.stop()
    print("saw_face:", seen)
    return 0


def cmd_mem(args) -> int:
    cfg = _load_cfg()
    from companion.llm.memory import MemoryStore

    mem = MemoryStore(cfg.abspath(cfg.memory.chroma_dir), cfg.memory.enabled, cfg.memory.top_k)
    if not mem.available:
        print("Memory unavailable (install mem0ai + chromadb)."); return 2
    if args.action == "search":
        q = " ".join(args.query)
        for m in mem.retrieve(q, args.speaker):
            print(" -", m)
    elif args.action == "add":
        mem.add(" ".join(args.query), args.speaker)
        print("added.")
    return 0


def cmd_speaker(args) -> int:
    cfg = _load_cfg()
    from companion.audio.io import AudioInput
    from companion.audio.speaker_id import SpeakerID
    import numpy as np

    sid = SpeakerID(cfg.abspath(cfg.speaker_id.model_path), cfg.abspath(cfg.speaker_id.speakers_file), cfg.speaker_id.match_threshold)
    if not sid.available:
        print("TitaNet unavailable — download the ONNX model first."); return 2
    ai = AudioInput({"sample_rate": cfg.audio.sample_rate, "channels": cfg.audio.channels, "chunk_size": cfg.audio.chunk_size, "input_device_name": cfg.audio.input_device_name})
    ai.start()
    print("Speak for 4 s...")
    t0 = time.time()
    chunks = []
    try:
        while time.time() - t0 < 4.0:
            c = ai.read(timeout=0.5)
            if c is not None:
                chunks.append(c)
    finally:
        ai.stop()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if args.action == "enrol":
        ok = sid.enrol(args.name, audio)
        print("enrolled" if ok else "failed")
    elif args.action == "id":
        name, score = sid.identify(audio)
        print(f"→ {name or 'unknown'}  (score={score:.3f})")
    elif args.action == "list":
        for n in sid.known_speakers():
            print(" -", n)
    return 0


def cmd_tools(args) -> int:
    cfg = _load_cfg()
    from companion.llm.function_gemma import FunctionGemma
    from companion.tools import registry

    registry.load_all_tools()
    fg = FunctionGemma(cfg.abspath(cfg.function_gemma.model_path), cfg.function_gemma.enabled, cfg.function_gemma.confidence_threshold)
    if fg.available:
        fg.set_tools(registry.all_schemas())
        call = fg.detect(" ".join(args.text))
        if call:
            print(f"tool: {call.name}({call.args})")
            print(" →", registry.invoke(call.name, call.args))
        else:
            print("No tool call detected.")
    else:
        print("FunctionGemma unavailable — listing registered tools instead:")
        for name in registry.all_schemas():
            print(" -", name["name"])
    return 0


def cmd_face(args) -> int:
    cfg = _load_cfg()
    from companion.display.renderer import make_renderer
    from companion.display.state import FaceState

    r = make_renderer(cfg.display)
    if r is None:
        print("No display backend available."); return 2
    r.set_action_callback(lambda n, p: print(f"action: {n} {p}"))
    r.start()
    presets = {
        "neutral":   FaceState(),
        "happy":     FaceState(valence=+0.7, arousal=+0.3),
        "excited":   FaceState(valence=+0.7, arousal=+0.7),
        "surprised": FaceState(valence=+0.1, arousal=+0.9),
        "calm":      FaceState(valence=+0.5, arousal=-0.4),
        "sad":       FaceState(valence=-0.6, arousal=-0.2),
        "angry":     FaceState(valence=-0.7, arousal=+0.6),
        "sleep":     FaceState(sleep=True),
        "talking":   FaceState(talking=True, valence=+0.2),
        "listening": FaceState(listening=True, valence=+0.1),
    }
    r.set_face(presets.get(args.preset, presets["neutral"]))
    time.sleep(args.seconds)
    r.stop()
    return 0


def cmd_pipeline(_args) -> int:
    print("Launching full pipeline — Ctrl-C to exit")
    os.execvp(sys.executable, [sys.executable, os.path.join(_ROOT, "main.py")])


def cmd_all(args) -> int:
    failures = []
    for name, fn, a in (
        ("env",     cmd_env,     argparse.Namespace()),
        ("audio",   cmd_audio,   argparse.Namespace(seconds=3.0)),
        ("stt",     cmd_stt,     argparse.Namespace()),
        ("vision",  cmd_vision,  argparse.Namespace(seconds=4.0)),
    ):
        print(f"\n=== {name} ===")
        try:
            rc = fn(a)
            if rc != 0:
                failures.append(name)
        except Exception as exc:
            print(f"{name} raised: {exc!r}")
            failures.append(name)
    print("\n── Summary ──")
    print(f"Failures: {failures or 'none'}")
    return 0 if not failures else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="tests.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env").set_defaults(fn=cmd_env)
    a = sub.add_parser("audio"); a.add_argument("--seconds", type=float, default=10.0); a.set_defaults(fn=cmd_audio)
    sub.add_parser("stt").set_defaults(fn=cmd_stt)
    a = sub.add_parser("llm"); a.add_argument("prompt", nargs="*"); a.set_defaults(fn=cmd_llm)
    a = sub.add_parser("vlm"); a.add_argument("question", nargs="*"); a.set_defaults(fn=cmd_vlm)
    a = sub.add_parser("tts"); a.add_argument("text", nargs="*"); a.set_defaults(fn=cmd_tts)
    a = sub.add_parser("vision"); a.add_argument("--seconds", type=float, default=10.0); a.set_defaults(fn=cmd_vision)
    a = sub.add_parser("mem")
    a.add_argument("action", choices=["search", "add"])
    a.add_argument("query", nargs="*")
    a.add_argument("--speaker", default="unknown")
    a.set_defaults(fn=cmd_mem)
    a = sub.add_parser("speaker")
    a.add_argument("action", choices=["enrol", "id", "list"])
    a.add_argument("--name", default="unknown")
    a.set_defaults(fn=cmd_speaker)
    a = sub.add_parser("tools"); a.add_argument("text", nargs="+"); a.set_defaults(fn=cmd_tools)
    a = sub.add_parser("face")
    a.add_argument("preset", choices=["neutral", "happy", "excited", "surprised",
                                       "calm", "sad", "angry", "sleep",
                                       "talking", "listening"])
    a.add_argument("--seconds", type=float, default=5.0)
    a.set_defaults(fn=cmd_face)
    sub.add_parser("pipeline").set_defaults(fn=cmd_pipeline)
    sub.add_parser("all").set_defaults(fn=cmd_all)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
