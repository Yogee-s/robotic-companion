"""Conversation Manager — orchestrates the full voice pipeline.

Pipeline:

    Audio → VAD → (+ EOU) → STT → Route → (LLM | VLM | Tool) → TTS → Speaker

Streaming — the LLM emits tokens; pysbd detects sentence boundaries;
completed sentences are synthesised by TTS and streamed straight into the
speaker's `aplay` process before the LLM has finished. First-audio
typically lands within 800 ms of end-of-speech.

Interruption — while speaking, a fresh VAD speech-start cancels the
in-flight LLM, tears down `aplay`, and resets to LISTENING within ~300 ms.

Context injection — emotion, scene caption, and per-speaker retrieved
memories are prepended to each user turn as `[user_emotion: …]`
`[scene: …]` `[remembered: …]` hints. Hints are gated to avoid noise
(see `companion/llm/prompt.py`).

Touchscreen — actions fired from the pygame or ESP32 display (e.g.
`stop_talking`, `mute_mic`, `timer`) are funnelled through
`handle_ui_action()` which reuses the same code paths as voice tools.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from companion.audio.eou import EndOfUtteranceDetector
from companion.audio.io import AudioInput, AudioOutput
from companion.audio.speaker_id import SpeakerID
from companion.audio.stt import SpeechToText
from companion.audio.tts import TextToSpeech
from companion.audio.vad import VoiceActivityDetector
from companion.core.config import AppConfig
from companion.llm.engine import LLMEngine
from companion.llm.function_gemma import FunctionGemma
from companion.llm.memory import MemoryStore
from companion.llm.prompt import (
    EmotionHint,
    build_system_prompt,
    format_emotion_hint,
    format_memory_hint,
    format_scene_hint,
    prepare_user_message,
)
from companion.llm.router import Route, decide_route
from companion.tools import registry as tool_registry
from companion.vision.pipeline import EmotionPipeline
from companion.vision.scene_watcher import SceneWatcher
from companion.vision.vlm import MoondreamVLM

log = logging.getLogger(__name__)


class ConversationState:
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# ── sentence tokenizer (pysbd if available, else fallback) ──────────────────
try:
    import pysbd  # type: ignore

    _SEG = pysbd.Segmenter(language="en", clean=False)

    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SEG.segment(text) if s.strip()]
except ImportError:
    import re

    _FALLBACK_RE = re.compile(r"(.+?[.!?])(?=\s|$)|(.+)$", re.DOTALL)

    def _split_sentences(text: str) -> list[str]:
        return [m.group(1) or m.group(2) for m in _FALLBACK_RE.finditer(text) if m]


class ConversationManager:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        tts: TextToSpeech,
        llm: LLMEngine,
        eou: Optional[EndOfUtteranceDetector] = None,
        emotion_pipeline: Optional[EmotionPipeline] = None,
        scene_watcher: Optional[SceneWatcher] = None,
        memory: Optional[MemoryStore] = None,
        speaker_id: Optional[SpeakerID] = None,
        vlm: Optional[MoondreamVLM] = None,
        function_gemma: Optional[FunctionGemma] = None,
    ) -> None:
        self.cfg = cfg
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._llm = llm
        self._eou = eou
        self._emotion = emotion_pipeline
        self._scene = scene_watcher
        self._memory = memory
        self._speaker_id = speaker_id
        self._vlm = vlm
        self._function_gemma = function_gemma

        conv = cfg.conversation
        self._max_history = conv.max_history
        self._verbosity = conv.verbosity
        self._log_enabled = conv.log_conversations
        self._log_dir = cfg.abspath(conv.log_directory)
        self._mode = conv.mode
        self._allow_interruption = conv.allow_interruption
        self._singlish = cfg.app.singlish

        # Hint + memory bookkeeping
        self._last_emotion_hint: Optional[EmotionHint] = None
        self._current_speaker: Optional[str] = None
        self._last_scene_emitted_at: float = 0.0
        self._last_scene_caption: str = ""

        # State
        self._state = ConversationState.IDLE
        self._history: list[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # GUI / display callbacks
        self.on_state_changed: Optional[Callable[[str], None]] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_response_token: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_audio_pcm: Optional[Callable[[bytes, int], None]] = None

        # VAD wiring
        self._vad.on_speech_start = self._on_speech_start
        self._vad.on_speech_end = self._on_speech_end

        if self._log_enabled:
            os.makedirs(self._log_dir, exist_ok=True)

        # Register tool schemas with FunctionGemma if present
        if self._function_gemma is not None:
            tool_registry.load_all_tools()
            self._function_gemma.set_tools(tool_registry.all_schemas())

    # ── state helpers ────────────────────────────────────────────────────
    def _set_state(self, s: str) -> None:
        if s == self._state:
            return
        old = self._state
        self._state = s
        log.info(f"State: {old} → {s}")
        if self.on_state_changed is not None:
            try:
                self.on_state_changed(s)
            except Exception:
                pass

    # ── public lifecycle ─────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._audio_input.start()
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        self._set_state(ConversationState.IDLE)
        log.info(f"Conversation started (mode={self._mode})")

    def stop(self) -> None:
        self._running = False
        self._audio_output.stop()
        self._audio_input.stop()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._set_state(ConversationState.IDLE)
        log.info("Conversation stopped")

    # ── main audio loop ──────────────────────────────────────────────────
    def _main_loop(self) -> None:
        while self._running:
            try:
                chunk = self._audio_input.read(timeout=0.5)
                if chunk is None:
                    continue

                # PTT guard
                if self._mode == "ptt" and self._state == ConversationState.IDLE:
                    continue

                # Interruption while speaking
                if self._state == ConversationState.SPEAKING:
                    if not self._allow_interruption:
                        continue
                    if self._audio_output.is_playing or self._audio_output.recently_played:
                        continue
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    if rms > 0.12:
                        self._interrupt()
                    continue

                if self._state == ConversationState.PROCESSING:
                    continue

                self._vad.process_chunk(chunk)
            except Exception as exc:
                if self._running:
                    log.error(f"Main loop error: {exc!r}")
                    time.sleep(0.1)

    def _interrupt(self) -> None:
        log.info("Interruption detected — cancelling generation")
        self._audio_output.stop()
        try:
            self._llm.cancel()
        except Exception:
            pass
        self._vad.reset()
        self._set_state(ConversationState.LISTENING)

    # ── VAD callbacks ────────────────────────────────────────────────────
    def _on_speech_start(self) -> None:
        if self._state in (ConversationState.IDLE, ConversationState.LISTENING):
            self._set_state(ConversationState.LISTENING)

    def _on_speech_end(self, audio: np.ndarray) -> None:
        if self._state != ConversationState.LISTENING:
            return
        threading.Thread(target=self._handle_turn, args=(audio,), daemon=True).start()

    # ── turn handler ────────────────────────────────────────────────────
    def _handle_turn(self, audio: np.ndarray) -> None:
        self._set_state(ConversationState.PROCESSING)
        t0 = time.time()
        try:
            # Transcribe
            user_text = self._stt.transcribe(audio)
            if not user_text or len(user_text.strip()) < 2:
                self._set_state(ConversationState.IDLE)
                return

            # Semantic end-of-turn — hold a little longer if user isn't done.
            if self._eou is not None and self._eou.available:
                if not self._eou.predict_end_of_turn(user_text):
                    log.debug("EOU: user seems unfinished — waiting extra window")
                    time.sleep(self._eou.extra_wait_ms / 1000.0)

            log.info(f"STT {time.time() - t0:.2f}s: \"{user_text}\"")
            if self.on_transcription is not None:
                try:
                    self.on_transcription(user_text)
                except Exception:
                    pass

            # Speaker ID — personalises memory + greeting
            if self._speaker_id is not None and self._speaker_id.available:
                name, score = self._speaker_id.identify(audio)
                if name:
                    self._current_speaker = name

            # Tool route? Run FunctionGemma first.
            tool_reply: Optional[str] = None
            has_tool_call = False
            if self._function_gemma is not None and self._function_gemma.available:
                call = self._function_gemma.detect(user_text)
                if call is not None:
                    has_tool_call = True
                    tool_reply = tool_registry.invoke(call.name, call.args)
                    log.info(f"Tool call: {call.name}({call.args}) → {tool_reply}")

            decision = decide_route(user_text, has_tool_call=has_tool_call)

            self._set_state(ConversationState.SPEAKING)

            # Dispatch by route
            if decision.route == Route.TOOL:
                ai_text = tool_reply or "Done."
                self._speak_text(ai_text)
            elif decision.route == Route.VQA:
                ai_text = self._handle_vqa(user_text)
                self._speak_text(ai_text)
            else:
                ai_text = self._handle_chat(user_text)

            if ai_text and self.on_response is not None:
                try:
                    self.on_response(ai_text)
                except Exception:
                    pass

            # Record + memory
            if ai_text:
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": ai_text})
                while len(self._history) > self._max_history * 2:
                    self._history.pop(0)
                if self._log_enabled:
                    self._append_jsonl(user_text, ai_text)
                if self._memory is not None:
                    # Persist the user turn (a concise fact) scoped by speaker
                    self._memory.add(f"User said: {user_text}", self._current_speaker or "unknown")
                    self._memory.add(
                        f"Assistant replied: {ai_text}", self._current_speaker or "unknown"
                    )
        except Exception as exc:
            log.error(f"Turn failed: {exc!r}")
            if self.on_error is not None:
                try:
                    self.on_error(str(exc))
                except Exception:
                    pass
        finally:
            self._vad.reset()
            if self._state != ConversationState.IDLE:
                self._set_state(ConversationState.IDLE)

    # ── route handlers ──────────────────────────────────────────────────
    def _handle_chat(self, user_text: str) -> str:
        # Build hints
        emotion_hint_str = self._build_emotion_hint()
        scene_hint_str = self._build_scene_hint()
        memory_hint_str = self._build_memory_hint(user_text)
        user_message = prepare_user_message(
            user_text,
            emotion_hint=emotion_hint_str,
            scene_hint=scene_hint_str,
            memory_hint=memory_hint_str,
        )
        system = build_system_prompt(
            self._llm.system_prompt,
            verbosity=self._verbosity,
            singlish=self._singlish,
            speaker_name=self._current_speaker,
        )

        # Streaming LLM → streaming TTS
        sentence_q: queue.Queue[str] = queue.Queue()
        generation_done = threading.Event()
        buf: list[str] = []

        def flush_if_sentence() -> None:
            full = "".join(buf)
            segments = _split_sentences(full)
            # Emit all but the last if there are multiple complete sentences;
            # keep the tail as work-in-progress.
            if len(segments) > 1:
                for s in segments[:-1]:
                    if s.strip():
                        sentence_q.put(s.strip())
                buf.clear()
                buf.append(segments[-1])

        def on_token(tok: str) -> None:
            buf.append(tok)
            if self.on_response_token is not None:
                try:
                    self.on_response_token(tok)
                except Exception:
                    pass
            flush_if_sentence()

        def stream_worker() -> None:
            opened = False
            while True:
                try:
                    sentence = sentence_q.get(timeout=0.15)
                except queue.Empty:
                    if generation_done.is_set() and sentence_q.empty():
                        break
                    continue
                if self._state != ConversationState.SPEAKING:
                    break
                pcm = self._tts.synthesize(sentence)
                if pcm is None:
                    continue
                if not opened:
                    self._audio_output.start_stream(self._tts.output_sample_rate)
                    opened = True
                self._audio_output.write_stream(pcm)
                if self.on_audio_pcm is not None:
                    try:
                        self.on_audio_pcm(pcm, self._tts.output_sample_rate)
                    except Exception:
                        pass
            if opened:
                self._audio_output.finish_stream()

        tstream = threading.Thread(target=stream_worker, daemon=True)
        tstream.start()

        ai_text = self._llm.generate(
            user_message=user_message,
            history=self._history[-self._max_history * 2 :],
            system_prompt=system,
            stream_callback=on_token,
        )
        # Flush tail
        tail = "".join(buf).strip()
        if tail:
            sentence_q.put(tail)
        generation_done.set()
        tstream.join(timeout=30)
        return ai_text

    def _handle_vqa(self, question: str) -> str:
        if self._vlm is None or not self._vlm.available:
            return "Sorry, I can't see right now."
        if self._emotion is None:
            return "Sorry, my camera isn't running."
        frame = self._emotion.get_state().frame
        if frame is None:
            return "Give me a moment, I can't see the scene clearly."
        answer = self._vlm.answer(frame, question)
        return answer or "I'm not sure what I'm looking at."

    def _speak_text(self, text: str) -> None:
        if not text.strip():
            return
        pcm = self._tts.synthesize(text)
        if pcm is None:
            return
        self._audio_output.start_stream(self._tts.output_sample_rate)
        self._audio_output.write_stream(pcm)
        self._audio_output.finish_stream()
        if self.on_audio_pcm is not None:
            try:
                self.on_audio_pcm(pcm, self._tts.output_sample_rate)
            except Exception:
                pass

    # ── hint builders ───────────────────────────────────────────────────
    def _build_emotion_hint(self) -> Optional[str]:
        if self._emotion is None:
            return None
        st = self._emotion.get_state()
        if not st.has_face:
            return None
        current = EmotionHint.from_state(st)
        hint = format_emotion_hint(current, self._last_emotion_hint)
        self._last_emotion_hint = current
        return hint

    def _build_scene_hint(self) -> Optional[str]:
        if self._scene is None:
            return None
        scene = self._scene.get_state()
        caption = scene.caption
        if not caption:
            return None
        now = time.time()
        if caption == self._last_scene_caption and now - self._last_scene_emitted_at < 20.0:
            return None
        self._last_scene_caption = caption
        self._last_scene_emitted_at = now
        return format_scene_hint(caption)

    def _build_memory_hint(self, user_text: str) -> Optional[str]:
        if self._memory is None or not self._memory.available:
            return None
        mems = self._memory.retrieve(user_text, self._current_speaker or "unknown")
        return format_memory_hint(mems)

    # ── conversation log (JSONL) ────────────────────────────────────────
    def _append_jsonl(self, user_text: str, ai_text: str) -> None:
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            path = os.path.join(self._log_dir, f"conversation_{date_str}.jsonl")
            entry = {
                "timestamp": datetime.now().isoformat(),
                "speaker": self._current_speaker or "unknown",
                "user": user_text,
                "assistant": ai_text,
                "emotion": (
                    self._emotion.get_state().to_dict() if self._emotion is not None else None
                ),
                "scene": (
                    self._scene.get_state().caption if self._scene is not None else None
                ),
            }
            with open(path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.debug(f"Log write failed: {exc!r}")

    # ── display / touchscreen action routing ────────────────────────────
    def handle_ui_action(self, name: str, payload: Optional[dict] = None) -> None:
        """Called by the display layer when the user taps a tile."""
        payload = payload or {}
        log.info(f"UI action: {name} {payload}")

        if name == "stop_talking":
            self._audio_output.stop()
            try:
                self._llm.cancel()
            except Exception:
                pass
            self._set_state(ConversationState.IDLE)
        elif name == "mute_mic":
            self._mode = "wake_word" if self._mode != "wake_word" else "ptt"
            log.info(f"Mic mode → {self._mode}")
        elif name == "sleep":
            self._set_state(ConversationState.IDLE)
        elif name in tool_registry._TOOLS:  # direct tool invocation
            result = tool_registry.invoke(name, payload)
            if result:
                self._speak_text(result)
        elif name == "restart":
            log.warning("UI requested restart — exiting cleanly for systemd to respawn.")
            self.stop()
        else:
            log.debug(f"Unhandled UI action: {name}")

    # ── push-to-talk ─────────────────────────────────────────────────────
    def push_to_talk_pressed(self) -> None:
        if self._mode != "ptt":
            return
        if self._state == ConversationState.SPEAKING:
            self._interrupt()
            return
        if self._state == ConversationState.IDLE:
            self._vad.reset()
            self._set_state(ConversationState.LISTENING)

    def push_to_talk_released(self) -> None:
        pass  # VAD handles end-of-speech

    # ── settings ─────────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        if mode not in ("ptt", "continuous", "wake_word"):
            return
        self._mode = mode
        log.info(f"Mode: {mode}")

    def set_push_to_talk(self, enabled: bool) -> None:
        self.set_mode("ptt" if enabled else "continuous")

    def set_singlish(self, enabled: bool) -> None:
        self._singlish = bool(enabled)

    def set_verbosity(self, level: str) -> None:
        if level in ("brief", "normal", "detailed"):
            self._verbosity = level

    def clear_history(self) -> None:
        self._history.clear()

    @property
    def singlish(self) -> bool:
        return self._singlish

    @property
    def verbosity(self) -> str:
        return self._verbosity

    @property
    def state(self) -> str:
        return self._state

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._running

    def get_history(self) -> list[dict]:
        return list(self._history)
