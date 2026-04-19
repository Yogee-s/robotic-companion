#!/usr/bin/env python3
"""Companion — application entry point.

Wires every subsystem together through a central EventBus and brings up
either the Qt main window or a headless run loop. Driven entirely by
`config.yaml` (+ optional `config.local.yaml` + env vars).

Run:
    python3 main.py

Graceful stop: Ctrl-C (systemd sends SIGTERM).

Bring-up order
--------------
1. Config + logging + readiness probe (fails loudly on missing models).
2. Core plumbing — EventBus.
3. Leaf subsystems (audio I/O, VAD, STT, TTS, LLM, vision, memory, tools).
4. Optional subsystems (motors, ESP32 renderer).
5. ConversationManager — the turn orchestrator.
6. BehaviorEngine + Coordinator + HealthMonitor — the coordination layer.
7. Qt window (or headless wait).

Shutdown is the reverse order with a 5 s hard timeout.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# cv2's wheel ships its own Qt plugin directory; if it ends up on the
# QT_QPA_PLATFORM_PLUGIN_PATH first, PyQt5's xcb plugin loader aborts
# the process with "could not load the Qt platform plugin xcb".
# Unset it before any Qt import so PyQt5's own plugin path wins.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from companion.core.config import load_config  # noqa: E402
from companion.core.logging import setup_logging  # noqa: E402
from companion.core.readiness import check_all  # noqa: E402

log = logging.getLogger("main")


def _banner(cfg) -> None:
    log.info("=" * 58)
    log.info("  Companion — multimodal AI companion robot")
    log.info(f"  LLM   : {cfg.llm.model}")
    log.info(f"  STT   : {cfg.stt.backend}")
    log.info(f"  TTS   : {cfg.tts.engine}")
    log.info(f"  mode  : {cfg.conversation.mode}")
    log.info(f"  disp  : {cfg.display.backend}")
    log.info(f"  motor : {'on' if cfg.motor.enabled else 'off'}"
             f"{' (sim)' if cfg.motor.sim_only else ''}")
    log.info("=" * 58)


# ─── subsystem construction helpers (one per logical stack) ────────────────

def _build_audio_stack(cfg):
    """Return (audio_in, audio_out, respeaker, vad, eou, stt, tts, speaker_id)."""
    from companion.audio.io import AudioInput, AudioOutput
    from companion.audio.respeaker import ReSpeakerArray
    from companion.audio.vad import VoiceActivityDetector
    from companion.audio.eou import EndOfUtteranceDetector
    from companion.audio.stt import SpeechToText
    from companion.audio.tts import TextToSpeech
    from companion.audio.speaker_id import SpeakerID

    audio_cfg = {
        "sample_rate": cfg.audio.sample_rate,
        "channels": cfg.audio.channels,
        "chunk_size": cfg.audio.chunk_size,
        "input_device_name": cfg.audio.input_device_name,
        "output_device_name": cfg.audio.output_device_name,
        "input_gain": cfg.audio.input_gain,
    }
    audio_in = AudioInput(audio_cfg)
    audio_out = AudioOutput({**audio_cfg, "output_sample_rate": cfg.tts.output_sample_rate})
    respeaker = ReSpeakerArray({
        "vendor_id": cfg.respeaker.vendor_id,
        "product_id": cfg.respeaker.product_id,
        "led_brightness": cfg.respeaker.led_brightness,
        "doa_enabled": cfg.respeaker.doa_enabled,
    })
    vad = VoiceActivityDetector({
        "threshold": cfg.vad.threshold,
        "silence_duration_ms": cfg.vad.silence_duration_ms,
        "min_speech_duration_ms": cfg.vad.min_speech_duration_ms,
        "speech_pad_ms": cfg.vad.speech_pad_ms,
    })
    eou = (
        EndOfUtteranceDetector(
            cfg.abspath(cfg.eou.model_path),
            cfg.eou.confidence_threshold,
            cfg.eou.extra_wait_ms,
        )
        if cfg.eou.enabled
        else None
    )
    stt = SpeechToText(cfg.stt, project_root=cfg.project_root)
    stt.warmup()
    tts = TextToSpeech(cfg.tts, project_root=cfg.project_root)
    speaker_id = (
        SpeakerID(
            cfg.abspath(cfg.speaker_id.model_path),
            cfg.abspath(cfg.speaker_id.speakers_file),
            cfg.speaker_id.match_threshold,
        )
        if cfg.speaker_id.enabled
        else None
    )
    return audio_in, audio_out, respeaker, vad, eou, stt, tts, speaker_id


def _build_llm_stack(cfg):
    """Load the main Gemma-4 LLM and create its memory store.

    FunctionGemma is NOT loaded here — it's a separate call so main.py
    can insert it after the vision stack, once the big Gemma allocation
    is done and ONNX arenas have stabilised.
    """
    from companion.llm.engine import LLMEngine
    from companion.llm.memory import MemoryStore

    llm = LLMEngine(
        cfg.llm,
        model_path=cfg.llm_model_path(),
        mmproj_path=cfg.abspath(cfg.llm.mmproj_path) if cfg.llm.mmproj_path else "",
    )
    llm.load()

    memory = MemoryStore(
        cfg.abspath(cfg.memory.chroma_dir),
        enabled=cfg.memory.enabled,
        top_k=cfg.memory.top_k,
        max_entries_per_speaker=cfg.memory.max_entries_per_speaker,
    )

    # FunctionGemma built later by `_build_function_gemma`. Return None
    # here so the existing unpack in main() stays a 3-tuple.
    return llm, memory, None


def _build_function_gemma(cfg):
    """Load FunctionGemma after all other GPU consumers.

    On an 8 GB unified-memory Jetson, the tool router is the last
    subsystem to grab VRAM: if there's GPU room left, it uses it; if
    not, `FunctionGemma._load` falls back to CPU automatically.
    """
    from companion.llm.function_gemma import FunctionGemma

    return FunctionGemma(
        cfg.abspath(cfg.function_gemma.model_path),
        enabled=cfg.function_gemma.enabled,
        confidence_threshold=cfg.function_gemma.confidence_threshold,
    )


def _build_vision_stack(cfg, llm, arbiter):
    """Return (emotion_pipeline, scene_watcher) — either may be None."""
    emotion_pipeline = None
    scene_watcher = None
    if not cfg.vision.enabled:
        return emotion_pipeline, scene_watcher

    from companion.vision.pipeline import EmotionPipeline
    from companion.vision.scene_watcher import SceneWatcher

    vision_cfg = {
        "sensor_id": cfg.vision.sensor_id,
        "width": cfg.vision.width,
        "height": cfg.vision.height,
        "fps": cfg.vision.fps,
        "flip_method": cfg.vision.flip_method,
        "use_csi": cfg.vision.use_csi,
        "detect_every_n_frames": cfg.vision.detect_every_n_frames,
        "yolo_pose_model_path": cfg.abspath(cfg.vision.yolo_pose_model_path),
        "emotion_model_path": cfg.abspath(cfg.vision.emotion_model_path),
        "emotion_enabled": cfg.vision.emotion_enabled,
        "face_score_threshold": cfg.vision.face_score_threshold,
        "smoothing": cfg.vision.smoothing,
        "staleness_fade_seconds": cfg.vision.staleness_fade_seconds,
    }
    emotion_pipeline = EmotionPipeline(vision_cfg)
    emotion_pipeline.start()

    # Scene captioning via the multimodal LLM (Moondream consolidated away)
    if getattr(llm, "is_multimodal", False):
        scene_watcher = SceneWatcher(
            llm,
            lambda: emotion_pipeline.get_state().frame,
            watch_hz=cfg.vlm.scene_watch_hz,
            arbiter=arbiter,
        )
        scene_watcher.start()
    return emotion_pipeline, scene_watcher


def _build_motor_stack(cfg):
    """Return (head, face_tracker) — both None when motor.enabled is false."""
    if not cfg.motor.enabled:
        return None, None
    from companion.motor.controller import HeadController
    from companion.vision.face_tracker import FaceTracker

    head = HeadController(cfg.motor)
    try:
        head.connect()
        head.enable_torque(True)
        if cfg.motor.home_on_startup:
            head.home()
    except Exception as exc:
        log.warning("HeadController init failed: %r — continuing without motors", exc)
        return None, None
    return head, FaceTracker  # tracker constructed later once we know the vision pipeline


def _build_renderer(cfg, conversation):
    from companion.display.renderer import make_renderer

    renderer = make_renderer(cfg.display)
    if renderer is not None:
        renderer.set_action_callback(
            lambda name, payload: conversation.handle_ui_action(name, payload)
        )
        renderer.start()
    return renderer


# ─── main ───────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_config(os.path.join(_HERE, "config.yaml"))
    setup_logging(cfg.app.log_level)
    # Must happen before any ONNX session is constructed — cached by the
    # provider probe on first use.
    if not cfg.runtime.onnx_cuda_enabled:
        os.environ["COMPANION_ONNX_CPU_ONLY"] = "1"
    _banner(cfg)

    # ── readiness gate ──────────────────────────────────────────────────
    rep = check_all(cfg)
    rep.print()
    if not rep.ok:
        log.error("Readiness check failed — aborting.")
        return 1

    # ── core plumbing ───────────────────────────────────────────────────
    from companion.core.event_bus import EventBus
    from companion.core.gpu_arbiter import GPUArbiter
    from companion.core.telemetry import TelemetryRecorder
    from companion.core.events import TurnCompleted

    bus = EventBus(async_workers=1)
    bus.start()
    arbiter = GPUArbiter()
    telemetry = TelemetryRecorder(
        log_dir=cfg.abspath(cfg.app.log_dir),
        ring_size=cfg.runtime.telemetry_ring_size,
    )
    bus.subscribe(TurnCompleted, lambda ev: telemetry.record(ev.trace))

    # ── leaf subsystems ────────────────────────────────────────────────
    # LOAD ORDER MATTERS on unified-memory Jetson. The main Gemma-4 needs
    # a single ~2 GB contiguous VRAM block. Every CUDA ONNX session
    # (VAD, EOU, Parakeet, Kokoro, YOLO, HSEmotion) creates its own
    # arena — allocating them first fragments the GPU heap and breaks
    # the big llama-cpp allocation. Load the LLM first, then audio,
    # then vision, then FunctionGemma (loaded inside _build_llm_stack
    # we split it out here so FunctionGemma gets the scraps).
    llm, memory, _ = _build_llm_stack(cfg)
    audio_in, audio_out, respeaker, vad, eou, stt, tts, speaker_id = _build_audio_stack(cfg)
    emotion_pipeline, scene_watcher = _build_vision_stack(cfg, llm, arbiter)
    function_gemma = _build_function_gemma(cfg)

    # Motor + face tracker (both optional — `head` is None if motors
    # are disabled or init failed, in which case FaceTracker is also None).
    head, _tracker_cls = _build_motor_stack(cfg)
    face_tracker = None
    if head is not None and emotion_pipeline is not None:
        from companion.vision.face_tracker import FaceTracker
        face_tracker = FaceTracker(
            head=head,
            vision=emotion_pipeline,
            kp=0.3,
            deadband_deg=4.0,
            update_hz=15.0,
            frame_width=cfg.vision.width,
            frame_height=cfg.vision.height,
        )

    # ── ConversationManager ─────────────────────────────────────────────
    from companion.conversation.manager import ConversationManager

    conversation = ConversationManager(
        cfg,
        audio_input=audio_in,
        audio_output=audio_out,
        vad=vad,
        stt=stt,
        tts=tts,
        llm=llm,
        eou=eou,
        emotion_pipeline=emotion_pipeline,
        scene_watcher=scene_watcher,
        memory=memory,
        speaker_id=speaker_id,
        function_gemma=function_gemma,
        event_bus=bus,
        respeaker=respeaker,
    )

    # Make SceneWatcher pause during active turns (it only polls the flag).
    if scene_watcher is not None:
        scene_watcher._is_turn_active = _turn_active_probe(conversation)

    # Tool notifiers (timer / reminder speak through the manager's TTS path).
    from companion.tools import remind_me as _remind_me
    from companion.tools import timer as _timer

    _timer.set_notifier(conversation._speak_text)
    _remind_me.set_notifier(conversation._speak_text)
    _remind_me.load_pending()

    # ── renderer (depends on the manager for touch action routing) ─────
    renderer = _build_renderer(cfg, conversation)

    # ── coordination layer ──────────────────────────────────────────────
    from companion.behavior.engine import BehaviorEngine
    from companion.conversation.coordinator import Coordinator
    from companion.core.health import HealthMonitor

    behavior = BehaviorEngine(
        renderer=renderer,
        emotion_pipeline=emotion_pipeline,
        respeaker=respeaker,
        face_tracker=face_tracker,
        event_bus=bus,
        tick_hz=cfg.runtime.behavior_tick_hz,
    )
    behavior.start()

    coordinator = Coordinator(event_bus=bus, conversation_manager=conversation)
    coordinator.start()

    health = HealthMonitor(
        event_bus=bus,
        tick_hz=cfg.runtime.health_tick_hz,
        audio_input=audio_in,
        emotion_pipeline=emotion_pipeline,
        renderer=renderer,
        head_controller=head,
    )
    health.start()

    # ── proactive engine (existing — no changes) ────────────────────────
    from companion.core.proactive import ProactiveEngine

    proactive = ProactiveEngine(cfg, conversation, emotion_pipeline, scene_watcher)
    proactive.start()

    # ── bring it up ─────────────────────────────────────────────────────
    conversation.start()

    # Startup audio check — speaks a short greeting so the user knows the
    # audio output path (TTS → ALSA → speaker) is working end-to-end before
    # any conversation happens. Fire-and-forget: any error is logged but
    # does not block startup.
    try:
        threading.Thread(
            target=conversation._speak_text,
            args=(None, "Hi, I'm ready."),
            daemon=True,
            name="startup-greeting",
        ).start()
    except Exception as exc:
        log.warning("Startup greeting failed: %r", exc)

    if getattr(respeaker, "is_connected", False):
        respeaker.start_doa_polling(
            callback=lambda ang, voice: None,  # BehaviorEngine polls DOA directly
            interval_ms=cfg.gui.doa_update_interval_ms,
        )

    # Shared shutdown closure so Qt and headless paths stay identical.
    subsystems = _ShutdownOrder(
        conversation=conversation,
        behavior=behavior,
        coordinator=coordinator,
        health=health,
        proactive=proactive,
        scene_watcher=scene_watcher,
        emotion_pipeline=emotion_pipeline,
        renderer=renderer,
        head=head,
        respeaker=respeaker,
        bus=bus,
    )

    return _run_ui_or_headless(cfg, conversation, emotion_pipeline, scene_watcher,
                               respeaker, subsystems)


# ─── UI / headless runners ─────────────────────────────────────────────────

def _run_ui_or_headless(cfg, conversation, emotion_pipeline, scene_watcher,
                         respeaker, subsystems) -> int:
    # Headless is the default. The ESP32 face is the primary UI; the
    # Qt debug window is an opt-in developer surface that tends to
    # collide with cv2's bundled Qt plugins on Jetson (xcb loader
    # abort()s at QApplication init — unrecoverable from Python).
    # Opt in with `COMPANION_GUI=1 python3 main.py`.
    want_gui = os.environ.get("COMPANION_GUI", "").lower() in ("1", "true", "yes")
    force_headless = os.environ.get("COMPANION_HEADLESS", "").lower() in ("1", "true", "yes")
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if force_headless or not want_gui or not has_display:
        if force_headless:
            reason = "COMPANION_HEADLESS set"
        elif not want_gui:
            reason = "COMPANION_GUI not set — Qt debug window opt-in only"
        else:
            reason = "no DISPLAY / WAYLAND_DISPLAY"
        log.info("Running headless (%s). Ctrl-C to quit.", reason)
        return _run_headless(subsystems)

    try:
        # Import cv2 first so its Qt libs resolve before PyQt5 loads —
        # otherwise the two conflict. Some cv2 wheels on Jetson ship
        # their own Qt plugins which collide with PyQt5's xcb loader.
        try:
            import cv2  # noqa: F401
        except Exception:
            pass

        from PyQt5.QtWidgets import QApplication  # type: ignore

        from companion.ui.main_window import MainWindow
        from companion.ui.theme import apply_theme

        try:
            app = QApplication(sys.argv)
        except Exception as exc:
            log.warning("QApplication init failed (%r) — running headless.", exc)
            return _run_headless(subsystems)

        apply_theme(app)
        window = MainWindow(
            cfg,
            conversation,
            emotion_pipeline=emotion_pipeline,
            scene_watcher=scene_watcher,
            respeaker=respeaker,
        )

        def _on_signal(_sig=None, _frame=None):
            log.info("Signal received — shutting down…")
            subsystems.shutdown()
            app.quit()

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        window.show()
        log.info("Ready. Walk into view and say something; touch the screen for controls.")
        code = app.exec_()
        subsystems.shutdown()
        return code
    except ImportError:
        log.warning("PyQt5 unavailable — running headless.")
        return _run_headless(subsystems)
    except Exception as exc:
        log.warning("Qt UI failed (%r) — running headless.", exc)
        return _run_headless(subsystems)


def _run_headless(subsystems) -> int:
    """Plain event loop — Ctrl-C exits cleanly.

    In headless mode we also read stdin lines as fake user utterances: type
    any text + Enter and it's injected through the real LLM → TTS path,
    bypassing VAD/STT. Lets you prove the back-half of the pipeline works
    independently while we debug mic/VAD issues.
    """
    log.info("Headless run ready. Ctrl-C to quit.")
    log.info("PTT: hold SPACE to talk. Release to send. Ctrl-C to quit.")
    stop_event = threading.Event()

    def _on_signal(_sig=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    conversation = subsystems._subs.get("conversation")

    def _stdin_injector():
        # Hold-to-talk on the spacebar. Uses termios cbreak mode so we can
        # detect individual key events instead of line-buffered input.
        # Sequence: first space → begin capture via AudioInput tap; we
        # keep capturing while Linux keyboard auto-repeat keeps sending
        # space characters. ~300ms without another space = released →
        # stop capture and feed the audio into `_on_speech_end()`, which
        # runs the full STT → LLM → TTS pipeline.
        import select
        import sys
        import termios
        import time
        import tty

        import numpy as np

        from companion.conversation.states import ConversationState

        # Linux key auto-repeat: ~500ms initial delay, then ~30ms interval.
        # Use a larger grace window right after press so a quick hold
        # doesn't look like a tap, and a tighter window for subsequent
        # repeats so release feels responsive.
        FIRST_GAP_S = 0.7
        REPEAT_GAP_S = 0.3
        MAX_RECORD_S = 12.0

        if not sys.stdin.isatty():
            log.info("stdin is not a TTY — spacebar PTT disabled.")
            return

        try:
            old_attrs = termios.tcgetattr(sys.stdin.fileno())
        except termios.error:
            log.info("Could not enter cbreak mode — spacebar PTT disabled.")
            return

        try:
            tty.setcbreak(sys.stdin.fileno())
            while not stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.25)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x03":  # Ctrl-C
                    stop_event.set()
                    return
                if ch != " ":
                    continue

                # Space pressed — begin capture.
                audio_in = conversation._audio_input
                chunks: list = []

                def _tap(chunk):
                    chunks.append(chunk.copy())

                audio_in.add_tap(_tap)
                record_start = time.time()
                last_space_ts = record_start
                log.info("PTT: listening (hold SPACE, release to send)")
                try:
                    conversation._set_state(ConversationState.LISTENING)
                except Exception:
                    pass

                try:
                    while True:
                        elapsed = time.time() - record_start
                        if elapsed > MAX_RECORD_S:
                            log.info("PTT: max duration reached")
                            break
                        gap = FIRST_GAP_S if elapsed < FIRST_GAP_S else REPEAT_GAP_S
                        remaining = max(0.0, gap - (time.time() - last_space_ts))
                        r2, _, _ = select.select([sys.stdin], [], [], remaining)
                        if not r2:
                            break  # released
                        c = sys.stdin.read(1)
                        if c == " ":
                            last_space_ts = time.time()
                        elif c == "\x03":
                            stop_event.set()
                            break
                finally:
                    audio_in.remove_tap(_tap)

                if not chunks:
                    log.warning("PTT: no audio captured (mic not delivering?)")
                    try:
                        conversation._set_state(ConversationState.IDLE_WATCHING)
                    except Exception:
                        pass
                    continue

                audio = np.concatenate(chunks).astype(np.float32)
                dur_s = len(audio) / 16000.0
                peak = float(np.max(np.abs(audio)))
                rms = float(np.sqrt(np.mean(audio ** 2)))
                log.info(
                    "PTT: captured %.2fs (rms=%.4f, peak=%.2f) — running STT+LLM+TTS",
                    dur_s, rms, peak,
                )
                conversation._on_speech_end(audio)
        finally:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, old_attrs
                )
            except Exception:
                pass

    threading.Thread(target=_stdin_injector, daemon=True, name="debug-stdin").start()

    stop_event.wait()
    subsystems.shutdown()
    return 0


# ─── shutdown ──────────────────────────────────────────────────────────────

class _ShutdownOrder:
    """Named tuple of subsystems + a single `shutdown()` that stops them in
    the right order with a hard timeout."""

    def __init__(self, **kw) -> None:
        self._subs = kw
        self._done = False

    def shutdown(self) -> None:
        if self._done:
            return
        self._done = True
        # Run shutdown in a watchdog thread so we can hard-exit if anything
        # hangs past the global deadline.
        deadline = 5.0

        def _run():
            try:
                self._subs["conversation"].stop()
            except Exception:
                pass
            for name in ("behavior", "coordinator", "health", "proactive",
                         "scene_watcher", "emotion_pipeline", "renderer"):
                sub = self._subs.get(name)
                if sub is None:
                    continue
                try:
                    sub.stop()
                except Exception:
                    pass
            head = self._subs.get("head")
            if head is not None:
                try:
                    head.home()
                    head.enable_torque(False)
                    head.disconnect()
                except Exception:
                    pass
            respeaker = self._subs.get("respeaker")
            if respeaker is not None:
                try:
                    respeaker.stop()
                except Exception:
                    pass
            bus = self._subs.get("bus")
            if bus is not None:
                try:
                    bus.stop()
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=deadline)
        if t.is_alive():
            log.warning("Shutdown exceeded %.1fs; hard-exiting.", deadline)
            os._exit(0)


def _turn_active_probe(conversation):
    """Return a zero-arg callable that reports whether a turn is in flight.
    Used by SceneWatcher to pause background VLM calls while the user is
    being served."""
    from companion.conversation.states import active_turn_states
    active = set(active_turn_states())

    def _probe() -> bool:
        return conversation.state in active

    return _probe


if __name__ == "__main__":
    sys.exit(main())
