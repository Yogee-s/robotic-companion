#!/usr/bin/env python3
"""
Jetson Orin Nano AI Companion — Main Entry Point.

Initializes all components and launches the voice conversation pipeline
with a PyQt5 GUI (or headless fallback).
"""

import logging
import os
import signal
import sys
import time

import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)


def setup_logging():
    try:
        import coloredlogs
        coloredlogs.install(
            level="INFO",
            fmt="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )


def load_config() -> dict:
    path = os.path.join(PROJECT_DIR, "config.yaml")
    if not os.path.exists(path):
        logging.warning(f"Config not found: {path}. Using defaults.")
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    logging.info(f"Config loaded from {path}")
    return cfg


def resolve_model_path(config: dict) -> dict:
    llm_cfg = config.get("llm", {})
    model_path = llm_cfg.get("model_path", "")
    if model_path and not os.path.isabs(model_path):
        llm_cfg["model_path"] = os.path.join(PROJECT_DIR, model_path)
    return config


def main():
    setup_logging()
    log = logging.getLogger("main")
    log.info("=" * 50)
    log.info("  Jetson Orin Nano AI Companion")
    log.info("=" * 50)

    config = resolve_model_path(load_config())

    # ── Initialize components ──
    log.info("Initializing components...")

    from companion.audio_io import AudioInput, AudioOutput
    audio_cfg = config.get("audio", {})
    audio_in = AudioInput(audio_cfg)
    audio_out = AudioOutput({
        **audio_cfg,
        "output_sample_rate": config.get("tts", {}).get("output_sample_rate", 22050),
    })
    log.info("  Audio I/O ready")

    from companion.respeaker import ReSpeakerArray
    respeaker = ReSpeakerArray(config.get("respeaker", {}))
    log.info(f"  ReSpeaker: {'connected' if respeaker.is_connected else 'simulated'}")

    from companion.vad import VoiceActivityDetector
    vad = VoiceActivityDetector(config.get("vad", {}))
    log.info("  VAD ready")

    from companion.stt import SpeechToText
    stt = SpeechToText(config.get("stt", {}))
    log.info(f"  STT: {stt.device_info}")
    log.info("  Warming up STT (first call is slow)...")
    stt.warmup()

    from companion.tts import TextToSpeech
    tts = TextToSpeech(config.get("tts", {}))
    log.info(f"  TTS: {tts.info}")

    from companion.llm import LLMEngine
    llm = LLMEngine(config.get("llm", {}))
    log.info("  Loading LLM (this may take a minute)...")
    llm.load()
    log.info(f"  LLM: {llm.info}")

    from companion.conversation import ConversationManager
    conversation = ConversationManager(
        config=config,
        audio_input=audio_in,
        audio_output=audio_out,
        vad=vad,
        stt=stt,
        tts=tts,
        llm=llm,
    )
    log.info("  Conversation manager ready")

    # ── Launch GUI ──
    from companion.gui import CompanionGUI, HAS_QT

    if not HAS_QT:
        log.warning("PyQt5 not available — running in headless mode.")
        _run_headless(conversation)
        return

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("AI Companion")

    window = CompanionGUI(
        config=config,
        conversation_manager=conversation,
        respeaker=respeaker,
    )
    window.connect_conversation(conversation)
    window.connect_audio_input(audio_in)
    window.set_model_info(f"{llm.info}\n{stt.device_info}\n{tts.info}")

    # DOA polling (via respeaker thread → Qt signal)
    if respeaker.is_connected:
        respeaker.start_doa_polling(
            callback=lambda a, v: window.doa_signal.emit(a, v),
            interval_ms=config.get("gui", {}).get("doa_update_interval_ms", 100),
        )

    conversation.start()
    log.info("Mode: Push-to-Talk (press SPACE to talk)")

    def shutdown(sig=None, frame=None):
        log.info("Shutting down...")
        conversation.stop()
        respeaker.stop()
        app.quit()

    signal.signal(signal.SIGINT, shutdown)

    window.show()
    log.info("AI Companion is ready! Start speaking.")
    exit_code = app.exec_()

    conversation.stop()
    respeaker.stop()
    log.info("Goodbye!")
    sys.exit(exit_code)


def _run_headless(conversation):
    """Terminal-only fallback when PyQt5 is not available."""
    import select
    import termios
    import tty

    log = logging.getLogger("headless")

    conversation.on_transcription = lambda t: print(f"\n  You: {t}")
    conversation.on_response = lambda t: print(f"  AI:  {t}\n")
    conversation.on_state_changed = lambda s: print(
        f"  [{s}]", end="\r", flush=True
    )

    conversation.start()
    print("\nAI Companion (headless, push-to-talk).")
    print("Hold SPACE to talk, release to stop.  Press 'q' to quit.\n")

    def shutdown(sig, frame):
        print("\nStopping...")
        conversation.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Set terminal to raw mode for key detection
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        space_held = False
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch == "q":
                    break
                if ch == " " and not space_held:
                    space_held = True
                    conversation.push_to_talk_pressed()
            else:
                if space_held:
                    space_held = False
                    conversation.push_to_talk_released()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        conversation.stop()


if __name__ == "__main__":
    main()
