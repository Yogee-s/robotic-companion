"""Conversation Manager — owns one turn at a time.

Pipeline per turn (all inside a single `Turn` object):

    mic → VAD → engagement gates → STT (streaming) → LLM (with prefill) → TTS → speaker

Key properties
--------------
* Continuous-mode listening with **face-gated VAD** — an utterance only
  triggers a turn if a face is present, the voiced speech lasted long
  enough, DOA (if available) lines up with the face, and the robot
  hasn't just finished speaking. Rejected utterances log at DEBUG.
* **One `Turn` per user utterance.** A new utterance arriving while the
  current turn is still running cancels the old one
  (`turn.cancel("new_utterance")`) so the user can barge-in during
  `CAPTURING_INTENT`/`THINKING`, not just during `SPEAKING`.
* **State is lock-guarded.** `_state_lock` wraps every read/write so the
  main loop and the turn worker can't race.
* **Event bus is the output channel.** Each phase publishes a typed
  event (`TurnStarted`, `TurnFirstToken`, `TurnFirstAudio`, `TurnCancelled`,
  `TurnCompleted`, `StateChanged`) on the shared `EventBus`. Legacy
  `on_state_changed` / `on_response` callbacks remain for the Qt window.

Touchscreen `handle_ui_action` routes the three foundational tiles:
Mute-mic (toggle), Stop (cancel in-flight), Sleep (soft idle) — plus the
More-list tiles (Volume, Restart) and any registered tool name.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from companion.audio.barge_in import BargeInDetector
from companion.audio.eou import EndOfUtteranceDetector
from companion.audio.io import AudioInput, AudioOutput
from companion.audio.speaker_id import SpeakerID
from companion.audio.stt import SpeechToText
from companion.audio.tts import TextToSpeech
from companion.audio.vad import VoiceActivityDetector
from companion.conversation.states import (
    ConversationState,
    assert_legal,
    active_turn_states,
)
from companion.conversation.turn import Turn
from companion.core.config import AppConfig
from companion.core.errors import CompanionError, LLMError, STTError, TTSError
from companion.core.events import (
    AffectTag as _AffectTagEvent,
    StateChanged,
    TurnCancelled,
    TurnCompleted,
    TurnFirstAudio,
    TurnFirstToken,
    TurnStarted,
    VisemeStream,
)
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

log = logging.getLogger(__name__)

# Short-opener instruction appended to the system prompt. Keeps the first
# sentence tight so TTS starts well under a second after end-of-speech.
_SHORT_OPENER_DIRECTIVE = (
    "Your first sentence is under 8 words. Never start with filler like "
    "'Well,' or 'So,'. You may elaborate in the sentences after."
)

_AFFECT_RE = re.compile(r"\s*\[affect:\s*([a-z_]+)\s*\]\s*$", re.IGNORECASE)
_MUMBLE_PROMPTS = ("Hmm?", "Sorry?", "Didn't catch that.")


# ── sentence tokenizer (pysbd if available, else fallback) ─────────────────

try:
    import pysbd  # type: ignore

    _SEG = pysbd.Segmenter(language="en", clean=False)

    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SEG.segment(text) if s.strip()]
except ImportError:
    _FALLBACK_RE = re.compile(r"(.+?[.!?])(?=\s|$)|(.+)$", re.DOTALL)

    def _split_sentences(text: str) -> list[str]:
        return [m.group(1) or m.group(2) for m in _FALLBACK_RE.finditer(text) if m]


class ConversationManager:
    """Owns the Turn lifecycle; publishes events; wraps legacy callbacks."""

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
        vlm=None,                          # kept for backward compat — unused
        function_gemma: Optional[FunctionGemma] = None,
        event_bus=None,
        respeaker=None,
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
        self._function_gemma = function_gemma
        self._bus = event_bus
        self._respeaker = respeaker

        conv = cfg.conversation
        self._max_history = conv.max_history
        self._verbosity = conv.verbosity
        self._log_enabled = conv.log_conversations
        self._log_dir = cfg.abspath(conv.log_directory)
        self._mode = conv.mode
        self._allow_interruption = conv.allow_interruption
        self._singlish = cfg.app.singlish
        self._engagement = conv.engagement

        # Hint + memory bookkeeping
        self._last_emotion_hint: Optional[EmotionHint] = None
        self._current_speaker: Optional[str] = None
        self._last_scene_emitted_at: float = 0.0
        self._last_scene_caption: str = ""
        self._last_face_seen_at: float = 0.0

        # State (protected by _state_lock)
        self._state = ConversationState.IDLE_WATCHING
        self._state_lock = threading.RLock()
        self._history: list[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Current in-flight turn + cancellation
        self._current_turn: Optional[Turn] = None
        self._turn_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, cfg.runtime.turn_workers),
            thread_name_prefix="turn",
        )

        # Barge-in / mute
        self._barge_in = BargeInDetector(vad)
        self._muted = False

        # Recent mumble bookkeeping (rate-limit the "hmm?" response)
        self._last_mumble_reply_at: float = 0.0

        # GUI / legacy callbacks
        self.on_state_changed: Optional[Callable[[str], None]] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_response_token: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_audio_pcm: Optional[Callable[[bytes, int], None]] = None

        # VAD wiring
        self._vad.on_speech_start = self._on_speech_start
        self._vad.on_speech_end = self._on_speech_end
        # Periodic "I hear you" diagnostic: log voiced probability peaks
        # at INFO so bring-up users can confirm the mic is delivering
        # audio and see where VAD sits relative to threshold. Rate-limited
        # so it doesn't flood the log.
        self._last_vad_peak_log_ts: float = 0.0
        self._vad.on_vad_probability = self._on_vad_probability

        if self._log_enabled:
            os.makedirs(self._log_dir, exist_ok=True)

        if self._function_gemma is not None:
            tool_registry.load_all_tools()
            self._function_gemma.set_tools(tool_registry.all_schemas())

    # ═══ public lifecycle ═══════════════════════════════════════════════
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._audio_input.start()
        self._thread = threading.Thread(
            target=self._main_loop, daemon=True, name="conversation-main"
        )
        self._thread.start()
        self._set_state(ConversationState.IDLE_WATCHING)
        log.info("Conversation started (mode=%s)", self._mode)

    def stop(self) -> None:
        self._running = False
        self._audio_output.stop()
        self._audio_input.stop()
        # Cancel the in-flight turn, if any
        with self._turn_lock:
            if self._current_turn is not None:
                self._current_turn.cancel("shutdown")
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._executor.shutdown(wait=False)
        self._set_state(ConversationState.IDLE_WATCHING)
        log.info("Conversation stopped")

    # ═══ state helpers ══════════════════════════════════════════════════
    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def _set_state(self, new: str) -> None:
        with self._state_lock:
            old = self._state
            if old == new:
                return
            assert_legal(old, new)
            self._state = new
        log.info("State: %s → %s", old, new)
        self._publish(StateChanged(old=old, new=new, timestamp=time.time()))
        if self.on_state_changed is not None:
            try:
                self.on_state_changed(new)
            except Exception:
                pass

    # ═══ main audio loop ════════════════════════════════════════════════
    def _main_loop(self) -> None:
        while self._running:
            try:
                chunk = self._audio_input.read(timeout=0.5)
                if chunk is None:
                    continue

                # Muted (touchscreen "Mute mic" active): don't process anything
                if self._muted:
                    continue

                state = self.state

                # Barge-in while speaking
                if state == ConversationState.SPEAKING:
                    if not self._allow_interruption:
                        continue
                    if self._audio_output.is_playing or self._audio_output.recently_played:
                        # Keep barge-in detector's noise floor updated even while speaking.
                        self._barge_in.should_interrupt(chunk)
                        continue
                    if self._barge_in.should_interrupt(chunk):
                        self._interrupt("barge_in")
                    continue

                # PTT mode: VAD is disabled. The audio tap in the stdin
                # PTT handler (main.py) captures chunks directly while
                # space is held and submits a turn on release. Feeding
                # VAD here would race with the PTT capture and risk
                # firing a second bogus turn mid-hold.
                if self._mode == "ptt":
                    continue

                # continuous / wake_word: fire VAD speech_start/end callbacks.
                self._vad.process_chunk(chunk)
            except Exception as exc:
                if self._running:
                    log.error("Main loop error: %r", exc)
                    time.sleep(0.1)

    # ═══ VAD callbacks ══════════════════════════════════════════════════
    def _on_vad_probability(self, prob: float) -> None:
        # Track the max prob in each 2-second window and log it
        # unconditionally. Lets bring-up users see whether Silero is
        # recognising their speech at all (pairs with the mic RMS log in
        # audio/io.py).
        if not hasattr(self, "_vad_window_peak"):
            self._vad_window_peak = 0.0
        if prob > self._vad_window_peak:
            self._vad_window_peak = prob
        now = time.time()
        if now - self._last_vad_peak_log_ts >= 2.0:
            self._last_vad_peak_log_ts = now
            log.info(
                "VAD max prob (2s window)=%.2f (threshold=%.2f)",
                self._vad_window_peak, self._vad.threshold,
            )
            self._vad_window_peak = 0.0

    def _on_speech_start(self) -> None:
        log.info("Speech START detected")
        state = self.state
        if state in (ConversationState.IDLE_WATCHING, ConversationState.LISTENING):
            self._set_state(ConversationState.LISTENING)

    def _on_speech_end(self, audio: np.ndarray) -> None:
        dur_ms = (len(audio) * 1000.0) / 16000.0
        log.info("Speech END (duration=%.0fms)", dur_ms)
        # Engagement gates decide whether this utterance enters a turn at all.
        if not self._engagement_gates_pass(audio):
            log.info("Utterance rejected by engagement gates")
            if self.state == ConversationState.LISTENING:
                self._set_state(ConversationState.IDLE_WATCHING)
            return

        # New utterance arrived — cancel an in-flight turn if one exists.
        with self._turn_lock:
            if self._current_turn is not None and not self._current_turn.is_cancelled:
                self._current_turn.cancel("new_utterance")

        turn = Turn()
        turn.audio = audio
        turn.mark("vad_end")
        with self._turn_lock:
            self._current_turn = turn
        self._executor.submit(self._handle_turn, turn)

    # ═══ engagement gates ═══════════════════════════════════════════════
    def _engagement_gates_pass(self, audio: np.ndarray) -> bool:
        """See EngagementConfig. All active gates must pass."""
        # Don't react to our own playback.
        if self._audio_output.is_playing or self._audio_output.recently_played:
            log.info("Engagement reject: own playback in progress")
            return False

        # Minimum voiced duration. `audio` is float32 at 16 kHz.
        sample_rate = 16000
        duration_ms = (len(audio) * 1000.0) / sample_rate
        if duration_ms < self._engagement.min_speech_ms:
            log.info(
                "Engagement reject: utterance %.0fms < min %dms",
                duration_ms, self._engagement.min_speech_ms,
            )
            return False

        if self._mode == "ptt":
            # In PTT mode the touchscreen/space bar already said "it's a turn"
            return True

        # Face presence (recently seen counts).
        if self._engagement.require_face and self._emotion is not None:
            now = time.time()
            state = self._emotion.get_state()
            if getattr(state, "has_face", False):
                self._last_face_seen_at = now
            lookback = self._engagement.face_lookback_ms / 1000.0
            if (now - self._last_face_seen_at) > lookback:
                return False

        # DOA-face concordance (soft — skip if unavailable).
        if (
            self._respeaker is not None
            and getattr(self._respeaker, "is_connected", False)
            and self._emotion is not None
        ):
            try:
                doa = float(self._respeaker.get_doa_signed())
            except Exception:
                doa = 0.0
            # If DOA exists and the face bbox is known, approximate the
            # face horizontal bearing as (bbox_x_center - frame_w/2) scaled
            # by a rough camera FOV. 62° HFOV matches the Jetson CSI cam.
            frame = getattr(self._emotion.get_state(), "frame", None)
            bbox = getattr(self._emotion.get_state(), "bbox", None)
            if frame is not None and bbox is not None:
                fw = frame.shape[1]
                cx = bbox[0] + bbox[2] / 2.0
                face_bearing_deg = ((cx - fw / 2.0) / (fw / 2.0)) * (62.0 / 2.0)
                if abs(face_bearing_deg - doa) > self._engagement.doa_face_concordance_deg:
                    log.debug(
                        "DOA-face mismatch: face=%.1f doa=%.1f", face_bearing_deg, doa
                    )
                    return False

        return True

    # ═══ turn handler ═══════════════════════════════════════════════════
    def _handle_turn(self, turn: Turn) -> None:
        """Runs on a turn worker thread. Owns STT → LLM → TTS for one turn."""
        self._set_state(ConversationState.CAPTURING_INTENT)
        try:
            if turn.is_cancelled:
                return

            # STT (uses streaming + prefill when Parakeet is primary)
            try:
                user_text = self._transcribe_with_prefill(turn)
            except Exception as exc:
                raise STTError(str(exc)) from exc

            turn.final_transcript = user_text
            turn.trace.transcript = user_text
            turn.mark("stt_final")

            if turn.is_cancelled:
                return

            if not user_text or len(user_text.strip()) < 3:
                self._maybe_say_mumble_reply()
                self._set_state(ConversationState.IDLE_WATCHING)
                return

            # Semantic end-of-utterance (optional extra wait)
            if self._eou is not None and self._eou.available:
                if not self._eou.predict_end_of_turn(user_text):
                    log.debug("EOU: user seems unfinished — waiting extra window")
                    if turn.cancel_event.wait(self._eou.extra_wait_ms / 1000.0):
                        return  # cancelled during the wait

            log.info("STT: %r", user_text)
            if self.on_transcription is not None:
                try:
                    self.on_transcription(user_text)
                except Exception:
                    pass

            # Speaker identification (once per turn)
            if self._speaker_id is not None and self._speaker_id.available and turn.audio is not None:
                try:
                    name, _score = self._speaker_id.identify(turn.audio)
                    if name:
                        self._current_speaker = name
                except Exception as exc:
                    log.debug("Speaker ID skipped: %r", exc)

            # Tool call?
            tool_reply: Optional[str] = None
            has_tool_call = False
            if self._function_gemma is not None and self._function_gemma.available:
                call = self._function_gemma.detect(user_text)
                if call is not None:
                    has_tool_call = True
                    turn.tool_name = call.name
                    try:
                        tool_reply = tool_registry.invoke(call.name, call.args)
                        log.info("Tool call: %s(%s) → %s", call.name, call.args, tool_reply)
                    except Exception as exc:
                        log.warning("Tool %s failed: %r", call.name, exc)
                        tool_reply = "Sorry, I couldn't do that right now."

            decision = decide_route(user_text, has_tool_call=has_tool_call)
            turn.route = decision.route.name.lower()
            turn.trace.route = turn.route

            # Announce turn start now that we know what we're doing.
            self._publish(TurnStarted(turn_id=turn.turn_id, transcript_preview=user_text[:80]))

            self._set_state(ConversationState.THINKING)

            if turn.is_cancelled:
                return

            self._set_state(ConversationState.SPEAKING)

            # Dispatch by route
            if decision.route == Route.TOOL:
                ai_text = tool_reply or "Done."
                self._speak_text(turn, ai_text)
            elif decision.route == Route.VQA:
                ai_text = self._handle_vqa(user_text)
                self._speak_text(turn, ai_text)
            else:
                ai_text = self._handle_chat(turn, user_text)

            if turn.is_cancelled:
                return

            # Strip and extract affect tag before emitting
            ai_text, affect = _extract_affect_tag(ai_text)
            if affect:
                self._publish(_AffectTagEvent(tag=affect))

            turn.reply_text = ai_text
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
                    self._memory.add(
                        f"User said: {user_text}", self._current_speaker or "unknown"
                    )
                    self._memory.add(
                        f"Assistant replied: {ai_text}", self._current_speaker or "unknown"
                    )
            turn.mark("completed")

        except CompanionError as exc:
            log.error("Turn %s failed: %s — %s", turn.turn_id, type(exc).__name__, exc)
            turn.trace.error_class = type(exc).__name__
            self._emit_error_speech(exc)
            if self.on_error is not None:
                try:
                    self.on_error(str(exc))
                except Exception:
                    pass
        except Exception as exc:
            log.exception("Turn %s unexpected failure: %r", turn.turn_id, exc)
            turn.trace.error_class = type(exc).__name__
            if self.on_error is not None:
                try:
                    self.on_error(str(exc))
                except Exception:
                    pass
        finally:
            self._vad.reset()
            if turn.is_cancelled:
                self._publish(TurnCancelled(turn_id=turn.turn_id, reason=turn.trace.interrupt_reason or "cancelled"))
            else:
                self._publish(TurnCompleted(turn_id=turn.turn_id, trace=turn.trace))
            with self._turn_lock:
                if self._current_turn is turn:
                    self._current_turn = None
            cur = self.state
            if cur not in (ConversationState.LISTENING,):
                self._set_state(ConversationState.IDLE_WATCHING)

    # ── STT with prefill ────────────────────────────────────────────────
    def _transcribe_with_prefill(self, turn: Turn) -> str:
        """Streaming STT when available; fires an LLM prefill on the first
        stable partial so prompt-eval overlaps with STT finalization.
        """
        if turn.audio is None:
            return ""

        final: str = ""

        def _partial_callback(text: str) -> None:
            nonlocal final
            if not text:
                return
            if turn.trace.t_stt_first_partial is None:
                turn.mark("stt_first_partial")
                # Stable first partial → kick off prefill in parallel.
                self._prefill_async(turn, text)
            # Keep the freshest partial as the running best-guess.
            final = text
            turn.partial_transcript = text

        if getattr(self._stt, "transcribe_stream", None) and getattr(self.cfg.stt, "streaming", True):
            audio_arr = turn.audio

            def _chunks():
                # Split the captured audio into ~250 ms slices so Parakeet's
                # streaming API receives it the same way it would in live
                # capture. For audio that VAD handed us as a single blob,
                # slicing is a pragmatic approximation.
                chunk_samples = int(0.25 * 16000)
                for i in range(0, len(audio_arr), chunk_samples):
                    if turn.is_cancelled:
                        return
                    yield audio_arr[i : i + chunk_samples]

            try:
                final = self._stt.transcribe_stream(_chunks(), _partial_callback)
            except Exception as exc:
                log.warning("Streaming STT failed, falling back to blocking: %r", exc)
                final = self._stt.transcribe(audio_arr)
        else:
            final = self._stt.transcribe(turn.audio)

        return final

    def _prefill_async(self, turn: Turn, partial: str) -> None:
        """Fire-and-forget KV-cache warm using the partial user text."""
        if turn.is_cancelled:
            return
        turn.mark("llm_prefill")

        def _task():
            try:
                system = build_system_prompt(
                    self._llm.system_prompt + " " + _SHORT_OPENER_DIRECTIVE,
                    verbosity=self._verbosity,
                    singlish=self._singlish,
                    speaker_name=self._current_speaker,
                )
                self._llm.prefill(
                    user_message=partial,
                    history=self._history[-self._max_history * 2 :],
                    system_prompt=system,
                )
            except Exception as exc:
                log.debug("Prefill skipped: %r", exc)

        # Use a short-lived thread (not the turn executor) so prefill can't
        # starve the main STT/LLM generation.
        threading.Thread(target=_task, daemon=True, name="llm-prefill").start()

    # ── CHAT route ──────────────────────────────────────────────────────
    def _handle_chat(self, turn: Turn, user_text: str) -> str:
        # Build hints in parallel with a tiny local pool — emotion / scene
        # / memory retrieval are all independent of each other.
        hint_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="hints")
        fut_em = hint_pool.submit(self._build_emotion_hint)
        fut_sc = hint_pool.submit(self._build_scene_hint)
        fut_me = hint_pool.submit(self._build_memory_hint, user_text)
        try:
            emotion_hint_str = fut_em.result(timeout=1.0)
            scene_hint_str = fut_sc.result(timeout=1.0)
            memory_hint_str = fut_me.result(timeout=1.0)
        except Exception as exc:
            log.debug("Hint assembly partial failure: %r", exc)
            emotion_hint_str = scene_hint_str = memory_hint_str = None
        finally:
            hint_pool.shutdown(wait=False)
            turn.mark("hints_ready")

        user_message = prepare_user_message(
            user_text,
            emotion_hint=emotion_hint_str,
            scene_hint=scene_hint_str,
            memory_hint=memory_hint_str,
        )
        system = build_system_prompt(
            self._llm.system_prompt + " " + _SHORT_OPENER_DIRECTIVE,
            verbosity=self._verbosity,
            singlish=self._singlish,
            speaker_name=self._current_speaker,
        )

        # Streaming LLM → streaming TTS via an in-turn sentence queue.
        sentence_q: queue.Queue[str] = queue.Queue()
        generation_done = threading.Event()
        buf: list[str] = []

        def flush_if_sentence() -> None:
            full = "".join(buf)
            segments = _split_sentences(full)
            if len(segments) > 1:
                for s in segments[:-1]:
                    if s.strip():
                        sentence_q.put(s.strip())
                buf.clear()
                buf.append(segments[-1])

        first_token_seen = [False]

        def on_token(tok: str) -> None:
            if turn.is_cancelled:
                self._llm.cancel()
                return
            if not first_token_seen[0]:
                first_token_seen[0] = True
                turn.mark("llm_first_token")
                self._publish(TurnFirstToken(turn_id=turn.turn_id, token=tok))
            buf.append(tok)
            if self.on_response_token is not None:
                try:
                    self.on_response_token(tok)
                except Exception:
                    pass
            flush_if_sentence()

        first_audio_seen = [False]

        def stream_worker() -> None:
            opened = False
            while True:
                if turn.is_cancelled:
                    break
                try:
                    sentence = sentence_q.get(timeout=0.15)
                except queue.Empty:
                    if generation_done.is_set() and sentence_q.empty():
                        break
                    continue
                if turn.is_cancelled or self.state != ConversationState.SPEAKING:
                    break
                try:
                    pcm = self._tts.synthesize(sentence)
                except Exception as exc:
                    raise TTSError(str(exc)) from exc
                if pcm is None:
                    continue
                if not opened:
                    self._audio_output.start_stream(self._tts.output_sample_rate)
                    opened = True
                self._audio_output.write_stream(pcm)
                if not first_audio_seen[0]:
                    first_audio_seen[0] = True
                    turn.mark("first_audio")
                    self._publish(TurnFirstAudio(turn_id=turn.turn_id))
                # Viseme stream for face lip-sync
                try:
                    from companion.audio import lip_sync  # local import
                    pcm_np = np.frombuffer(pcm, dtype=np.int16)
                    events = lip_sync.visemes_from_pcm(pcm_np, self._tts.output_sample_rate)
                    if events:
                        self._publish(VisemeStream(
                            turn_id=turn.turn_id,
                            events=list(events),
                            sample_rate=self._tts.output_sample_rate,
                            timestamp=time.time(),
                        ))
                except Exception:
                    pass
                # Feed the PCM into the barge-in detector so it can
                # subtract self-echo from mic input.
                try:
                    self._barge_in.note_tts_sample(pcm_np, self._tts.output_sample_rate)
                except Exception:
                    pass
                if self.on_audio_pcm is not None:
                    try:
                        self.on_audio_pcm(pcm, self._tts.output_sample_rate)
                    except Exception:
                        pass
            if opened:
                self._audio_output.finish_stream()
                turn.mark("audio_end")

        tstream = threading.Thread(target=stream_worker, daemon=True, name="tts-stream")
        tstream.start()

        try:
            ai_text = self._llm.generate(
                user_message=user_message,
                history=self._history[-self._max_history * 2 :],
                system_prompt=system,
                stream_callback=on_token,
            )
        except Exception as exc:
            raise LLMError(str(exc)) from exc

        tail = "".join(buf).strip()
        if tail:
            sentence_q.put(tail)
        generation_done.set()
        tstream.join(timeout=30)
        return ai_text or ""

    # ── VQA route ───────────────────────────────────────────────────────
    def _handle_vqa(self, question: str) -> str:
        """Visual Q&A via the multimodal LLM (consolidated from Moondream)."""
        if not getattr(self._llm, "is_multimodal", False):
            return "Sorry, I can't see right now."
        if self._emotion is None:
            return "Sorry, my camera isn't running."
        frame = self._emotion.get_state().frame
        if frame is None:
            return "Give me a moment, I can't see the scene clearly."
        try:
            answer = self._llm.answer(frame, question)
        except Exception:
            return "I'm not sure what I'm looking at."
        return answer or "I'm not sure what I'm looking at."

    # ── short-form speak (tools, mumble replies, degradation messages) ──
    def _speak_text(self, turn: Optional[Turn], text: str) -> None:
        if not text or not text.strip():
            return
        try:
            pcm = self._tts.synthesize(text)
        except Exception as exc:
            log.warning("TTS synth failed: %r", exc)
            pcm = None
        if pcm is None:
            return
        self._audio_output.start_stream(self._tts.output_sample_rate)
        self._audio_output.write_stream(pcm)
        if turn is not None and turn.trace.t_first_audio is None:
            turn.mark("first_audio")
            self._publish(TurnFirstAudio(turn_id=turn.turn_id))
        self._audio_output.finish_stream()
        if turn is not None:
            turn.mark("audio_end")
        if self.on_audio_pcm is not None:
            try:
                self.on_audio_pcm(pcm, self._tts.output_sample_rate)
            except Exception:
                pass

    def _maybe_say_mumble_reply(self) -> None:
        """Rotating, rate-limited 'Hmm?' on a short / unintelligible STT result."""
        now = time.time()
        if now - self._last_mumble_reply_at < 10.0:
            return
        self._last_mumble_reply_at = now
        idx = int(now) % len(_MUMBLE_PROMPTS)
        self._set_state(ConversationState.SPEAKING)
        self._speak_text(None, _MUMBLE_PROMPTS[idx])

    def _emit_error_speech(self, exc: CompanionError) -> None:
        """User-facing recovery speech matched to the error type."""
        msg = {
            "STTError": "Sorry, I didn't catch that. Could you say it again?",
            "LLMError": "Sorry, I lost my train of thought. Could you say that again?",
            "TTSError": "",  # silent — we can't speak anyway
            "ToolError": "I couldn't do that just now.",
            "ToolNetworkError": "I couldn't reach that service right now.",
            "VisionError": "",
            "MotorError": "",
            "SerialError": "",
        }.get(type(exc).__name__, "Something went wrong — let's try again.")
        if msg:
            try:
                self._speak_text(None, msg)
            except Exception:
                pass

    # ── interrupt / cancellation ────────────────────────────────────────
    def _interrupt(self, reason: str = "user_interrupt") -> None:
        log.info("Interrupt: %s", reason)
        self._audio_output.stop()
        with self._turn_lock:
            if self._current_turn is not None:
                self._current_turn.cancel(reason)
        try:
            self._llm.cancel()
        except Exception:
            pass
        self._vad.reset()
        self._set_state(ConversationState.LISTENING)

    # ═══ hint builders ══════════════════════════════════════════════════
    def _build_emotion_hint(self) -> Optional[str]:
        if self._emotion is None:
            return None
        st = self._emotion.get_state()
        if not getattr(st, "has_face", False):
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

    # ═══ JSONL conversation log ═════════════════════════════════════════
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
                    self._emotion.get_state().to_dict()
                    if self._emotion is not None
                    else None
                ),
                "scene": (
                    self._scene.get_state().caption if self._scene is not None else None
                ),
            }
            with open(path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.debug("Log write failed: %r", exc)

    # ═══ touchscreen / UI actions ═══════════════════════════════════════
    def handle_ui_action(self, name: str, payload: Optional[dict] = None) -> None:
        """Called by the renderer when the user taps a tile."""
        payload = payload or {}
        log.info("UI action: %s %s", name, payload)

        if name == "stop_talking":
            self._interrupt("ui_stop")
        elif name == "mute_mic":
            # Toggle between "continuous listening" and "muted".
            self._muted = not self._muted
            log.info("Mic muted: %s", self._muted)
        elif name == "sleep":
            self._muted = False  # wake the mic if user re-taps to leave sleep
            # Soft idle — the BehaviorEngine listens to state changes and
            # dims the face in IDLE_WATCHING anyway.
            self._interrupt("ui_sleep")
            self._set_state(ConversationState.IDLE_WATCHING)
        elif name in tool_registry._TOOLS:
            # Direct tool invocation from touchscreen (e.g. "volume")
            try:
                result = tool_registry.invoke(name, payload)
            except Exception as exc:
                log.warning("Tool %s failed via UI: %r", name, exc)
                result = "Sorry, I couldn't do that."
            if result:
                self._speak_text(None, result)
        elif name == "restart":
            log.warning("UI requested restart — stopping cleanly for systemd to respawn.")
            self.stop()
        else:
            log.debug("Unhandled UI action: %s", name)

    # ═══ PTT (for touchscreen / keyboard) ═══════════════════════════════
    def push_to_talk_pressed(self) -> None:
        if self._mode != "ptt":
            return
        state = self.state
        if state == ConversationState.SPEAKING:
            self._interrupt("ptt_press")
            return
        if state == ConversationState.IDLE_WATCHING:
            self._vad.reset()
            self._set_state(ConversationState.LISTENING)

    def push_to_talk_released(self) -> None:
        pass  # VAD handles end-of-speech

    # ═══ settings (Qt window writes these) ══════════════════════════════
    def set_mode(self, mode: str) -> None:
        if mode not in ("ptt", "continuous", "wake_word"):
            return
        self._mode = mode
        log.info("Mode: %s", mode)

    def set_push_to_talk(self, enabled: bool) -> None:
        self.set_mode("ptt" if enabled else "continuous")

    def set_singlish(self, enabled: bool) -> None:
        self._singlish = bool(enabled)

    def set_verbosity(self, level: str) -> None:
        if level in ("brief", "normal", "detailed"):
            self._verbosity = level

    def clear_history(self) -> None:
        self._history.clear()

    # ═══ read-only properties ═══════════════════════════════════════════
    @property
    def singlish(self) -> bool:
        return self._singlish

    @property
    def verbosity(self) -> str:
        return self._verbosity

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def muted(self) -> bool:
        return self._muted

    def get_history(self) -> list[dict]:
        return list(self._history)

    # ═══ internals ══════════════════════════════════════════════════════
    def _publish(self, event: Any) -> None:
        if self._bus is not None:
            self._bus.publish(event)


# ── affect tag parsing ──────────────────────────────────────────────────────

def _extract_affect_tag(text: str) -> tuple[str, Optional[str]]:
    """Strip a trailing `[affect: xxx]` marker from the LLM reply and
    return `(clean_text, tag_name_or_None)`."""
    if not text:
        return text, None
    m = _AFFECT_RE.search(text)
    if not m:
        return text, None
    tag = m.group(1).strip().lower()
    clean = _AFFECT_RE.sub("", text).strip()
    return clean, tag
