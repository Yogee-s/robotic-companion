#!/usr/bin/env python3
"""Companion — application entry point.

Wires every subsystem together, brings up the Qt main window (or
headless fallback), and handles clean shutdown. Everything is driven by
`config.yaml` — no flags or env vars.

Run:
    python3 main.py

Graceful stop: Ctrl-C.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from companion.core.config import load_config  # noqa: E402
from companion.core.logging import setup_logging  # noqa: E402

log = logging.getLogger("main")


def _banner(cfg) -> None:
    log.info("=" * 58)
    log.info("  Companion — multimodal AI companion robot")
    log.info(f"  LLM  : {cfg.llm.model}")
    log.info(f"  STT  : {cfg.stt.backend}")
    log.info(f"  TTS  : {cfg.tts.engine}")
    log.info(f"  VLM  : {'on' if cfg.vlm.enabled else 'off'}")
    log.info(f"  mode : {cfg.conversation.mode}")
    log.info(f"  disp : {cfg.display.backend}")
    log.info("=" * 58)


def main() -> int:
    cfg = load_config(os.path.join(_HERE, "config.yaml"))
    setup_logging(cfg.app.log_level)
    _banner(cfg)

    # ── audio stack ─────────────────────────────────────────────────────
    from companion.audio.io import AudioInput, AudioOutput
    from companion.audio.respeaker import ReSpeakerArray
    from companion.audio.vad import VoiceActivityDetector
    from companion.audio.eou import EndOfUtteranceDetector
    from companion.audio.stt import SpeechToText
    from companion.audio.tts import TextToSpeech
    from companion.audio.speaker_id import SpeakerID

    audio_cfg_dict = {
        "sample_rate": cfg.audio.sample_rate,
        "channels": cfg.audio.channels,
        "chunk_size": cfg.audio.chunk_size,
        "input_device_name": cfg.audio.input_device_name,
        "output_device_name": cfg.audio.output_device_name,
    }
    audio_in = AudioInput(audio_cfg_dict)
    audio_out = AudioOutput({**audio_cfg_dict, "output_sample_rate": cfg.tts.output_sample_rate})
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
    eou = EndOfUtteranceDetector(
        cfg.abspath(cfg.eou.model_path),
        cfg.eou.confidence_threshold,
        cfg.eou.extra_wait_ms,
    ) if cfg.eou.enabled else None

    stt = SpeechToText(cfg.stt, project_root=cfg.project_root)
    stt.warmup()
    tts = TextToSpeech(cfg.tts, project_root=cfg.project_root)

    speaker_id = SpeakerID(
        cfg.abspath(cfg.speaker_id.model_path),
        cfg.abspath(cfg.speaker_id.speakers_file),
        cfg.speaker_id.match_threshold,
    ) if cfg.speaker_id.enabled else None

    # ── LLM + VLM + memory + function-gemma ─────────────────────────────
    from companion.llm.engine import LLMEngine
    from companion.llm.function_gemma import FunctionGemma
    from companion.llm.memory import MemoryStore

    llm = LLMEngine(cfg.llm, model_path=cfg.llm_model_path())
    llm.load()

    vlm = None
    if cfg.vlm.enabled:
        from companion.vision.vlm import MoondreamVLM

        vlm = MoondreamVLM(
            cfg.abspath(cfg.vlm.model_path),
            cfg.abspath(cfg.vlm.mmproj_path),
            enabled=True,
            max_tokens=cfg.vlm.max_tokens,
        )

    memory = MemoryStore(
        cfg.abspath(cfg.memory.chroma_dir),
        enabled=cfg.memory.enabled,
        top_k=cfg.memory.top_k,
        max_entries_per_speaker=cfg.memory.max_entries_per_speaker,
    )

    function_gemma = FunctionGemma(
        cfg.abspath(cfg.function_gemma.model_path),
        enabled=cfg.function_gemma.enabled,
        confidence_threshold=cfg.function_gemma.confidence_threshold,
    )

    # ── vision pipeline + scene watcher ─────────────────────────────────
    emotion_pipeline = None
    scene_watcher = None
    if cfg.vision.enabled:
        from companion.vision import EmotionPipeline

        vision_cfg = {
            "sensor_id": cfg.vision.sensor_id,
            "width": cfg.vision.width,
            "height": cfg.vision.height,
            "fps": cfg.vision.fps,
            "flip_method": cfg.vision.flip_method,
            "use_csi": cfg.vision.use_csi,
            "detect_every_n_frames": cfg.vision.detect_every_n_frames,
            "face_model_path": cfg.abspath(cfg.vision.face_model_path),
            "emotion_model_path": cfg.abspath(cfg.vision.emotion_model_path),
            "face_score_threshold": cfg.vision.face_score_threshold,
            "smoothing": cfg.vision.smoothing,
            "staleness_fade_seconds": cfg.vision.staleness_fade_seconds,
        }
        emotion_pipeline = EmotionPipeline(vision_cfg)
        emotion_pipeline.start()

        if vlm is not None and vlm.available:
            from companion.vision.scene_watcher import SceneWatcher

            scene_watcher = SceneWatcher(
                vlm, lambda: emotion_pipeline.get_state().frame, watch_hz=cfg.vlm.scene_watch_hz
            )
            scene_watcher.start()

    # ── conversation manager ───────────────────────────────────────────
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
        vlm=vlm,
        function_gemma=function_gemma,
    )

    # Hook background tool notifiers so timers/reminders speak through TTS
    from companion.tools import remind_me as _remind_me
    from companion.tools import timer as _timer

    _timer.set_notifier(conversation._speak_text)
    _remind_me.set_notifier(conversation._speak_text)
    _remind_me.load_pending()

    # ── display layer ──────────────────────────────────────────────────
    from companion.display.renderer import make_renderer
    from companion.display.state import ConversationalState
    from companion.display.expressions import emotion_to_face

    renderer = make_renderer(cfg.display)
    if renderer is not None:
        renderer.set_action_callback(
            lambda name, payload: conversation.handle_ui_action(name, payload)
        )
        renderer.start()

    # Drive the face in a light background loop: polls state + pushes FaceState.
    def _face_loop():
        import time as _t

        while True:
            if renderer is None:
                return
            try:
                if emotion_pipeline is not None:
                    em = emotion_pipeline.get_state()
                else:
                    from companion.vision.pipeline import EmotionState
                    em = EmotionState()
                doa = respeaker.get_doa() if respeaker.is_connected else 0.0
                cmap = {
                    "idle": ConversationalState.IDLE,
                    "listening": ConversationalState.LISTENING,
                    "processing": ConversationalState.THINKING,
                    "speaking": ConversationalState.SPEAKING,
                }
                cs = cmap.get(conversation.state, ConversationalState.IDLE)
                fs = emotion_to_face(em, cs, doa_angle_deg=doa)
                renderer.set_face(fs)
            except Exception:
                pass
            _t.sleep(1.0 / 30.0)

    threading.Thread(target=_face_loop, daemon=True).start()

    # ── proactive engine ───────────────────────────────────────────────
    from companion.core.proactive import ProactiveEngine

    proactive = ProactiveEngine(cfg, conversation, emotion_pipeline, scene_watcher)
    proactive.start()

    # ── start conversation + bring up GUI ──────────────────────────────
    conversation.start()

    if respeaker.is_connected:
        respeaker.start_doa_polling(
            callback=lambda ang, voice: None,
            interval_ms=cfg.gui.doa_update_interval_ms,
        )

    try:
        from PyQt5.QtWidgets import QApplication  # type: ignore

        from companion.ui.main_window import MainWindow
        from companion.ui.theme import apply_theme

        app = QApplication(sys.argv)
        apply_theme(app)
        window = MainWindow(
            cfg,
            conversation,
            emotion_pipeline=emotion_pipeline,
            scene_watcher=scene_watcher,
            respeaker=respeaker,
        )

        def shutdown(_sig=None, _frame=None):
            log.info("Shutting down…")
            conversation.stop()
            proactive.stop()
            if scene_watcher is not None:
                scene_watcher.stop()
            if emotion_pipeline is not None:
                emotion_pipeline.stop()
            if renderer is not None:
                renderer.stop()
            respeaker.stop()
            app.quit()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        window.show()
        log.info("Ready. Press SPACE to talk.")
        code = app.exec_()
        return code
    except ImportError:
        log.warning("PyQt5 unavailable — running headless; Ctrl-C to quit.")
        stop_event = threading.Event()

        def shutdown(_sig=None, _frame=None):
            stop_event.set()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        stop_event.wait()
        conversation.stop()
        proactive.stop()
        if scene_watcher is not None:
            scene_watcher.stop()
        if emotion_pipeline is not None:
            emotion_pipeline.stop()
        if renderer is not None:
            renderer.stop()
        respeaker.stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())
