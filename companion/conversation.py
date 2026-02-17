"""
Conversation Manager.

Orchestrates the voice pipeline: Audio → VAD → STT → LLM → TTS → Speaker.

Key optimization: sentences are synthesized and played while the LLM is
still generating, so the user hears the first sentence much sooner.
"""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from companion.audio_io import AudioInput, AudioOutput
from companion.llm import LLMEngine
from companion.stt import SpeechToText
from companion.tts import TextToSpeech
from companion.vad import VoiceActivityDetector, VADState

logger = logging.getLogger(__name__)


class ConversationState:
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class ConversationManager:
    """
    Full voice conversation pipeline with streaming TTS.

    State flow: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
    During SPEAKING, the LLM may still be generating tokens while earlier
    sentences are already being spoken aloud.
    """

    def __init__(
        self,
        config: dict,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        tts: TextToSpeech,
        llm: LLMEngine,
    ):
        self._config = config
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._llm = llm

        conv_cfg = config.get("conversation", {})
        self._max_history = conv_cfg.get("max_history", 6)
        self._verbosity = conv_cfg.get("verbosity", "normal")
        self._log_conversations = conv_cfg.get("log_conversations", True)
        self._log_directory = conv_cfg.get("log_directory", "logs")

        # Mode: "ptt" (push-to-talk, default) or "continuous"
        self._mode = conv_cfg.get("mode", "ptt")
        self._push_to_talk = (self._mode == "ptt")
        self._singlish = False

        self._state = ConversationState.IDLE
        self._history: list[dict] = []
        self._running = False
        self._thread = None

        # GUI callbacks (set externally)
        self.on_state_changed: Optional[Callable[[str], None]] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_response_token: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Wire VAD callbacks
        self._vad.on_speech_start = self._on_speech_start
        self._vad.on_speech_end = self._on_speech_end

        if self._log_conversations:
            os.makedirs(self._log_directory, exist_ok=True)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state: str):
        old = self._state
        self._state = new_state
        logger.info(f"State: {old} → {new_state}")
        if self.on_state_changed:
            try:
                self.on_state_changed(new_state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Main audio loop
    # ------------------------------------------------------------------

    def start(self):
        """Start the conversation loop."""
        if self._running:
            return
        self._running = True
        self._audio_input.start()
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        self._set_state(ConversationState.IDLE)
        logger.info("Conversation manager started.")

    def _main_loop(self):
        while self._running:
            try:
                chunk = self._audio_input.read(timeout=0.5)
                if chunk is None:
                    continue

                # Push-to-talk guard
                if self._push_to_talk and self._state == ConversationState.IDLE:
                    continue

                # Interruption detection while speaking
                if self._state == ConversationState.SPEAKING:
                    # In PTT mode: never interrupt — let the AI finish.
                    if self._mode == "ptt":
                        continue
                    # In continuous mode: only check when speaker is silent
                    # and echo has faded (0.5 s cooldown).
                    if self._audio_output.is_playing or self._audio_output.recently_played:
                        continue
                    rms = np.sqrt(np.mean(chunk ** 2))
                    if rms > 0.12:
                        self._handle_interruption()
                    continue

                # Skip VAD while LLM/TTS are running
                if self._state == ConversationState.PROCESSING:
                    continue

                self._vad.process_chunk(chunk)

            except Exception as e:
                if self._running:
                    logger.error(f"Main loop error: {e}")
                    time.sleep(0.1)

    # ------------------------------------------------------------------
    # VAD callbacks
    # ------------------------------------------------------------------

    def _on_speech_start(self):
        if self._state in (ConversationState.IDLE, ConversationState.LISTENING):
            self._set_state(ConversationState.LISTENING)

    def _on_speech_end(self, audio: np.ndarray):
        if self._state != ConversationState.LISTENING:
            return
        threading.Thread(
            target=self._process_speech, args=(audio,), daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Core pipeline: STT → streaming LLM → streaming TTS → Play
    # ------------------------------------------------------------------

    def _process_speech(self, audio: np.ndarray):
        """
        Full pipeline with sentence-level streaming.

        While the LLM generates tokens, completed sentences are immediately
        synthesized and played, so the user hears the response sooner.
        """
        self._set_state(ConversationState.PROCESSING)
        t_start = time.time()

        try:
            # ── 1. Speech-to-Text ──
            user_text = self._stt.transcribe(audio)
            t_stt = time.time() - t_start

            if not user_text or len(user_text.strip()) < 2:
                logger.debug("Empty transcription, returning to idle.")
                self._set_state(ConversationState.IDLE)
                return

            logger.info(f"STT ({t_stt:.2f}s): \"{user_text}\"")
            if self.on_transcription:
                try:
                    self.on_transcription(user_text)
                except Exception:
                    pass

            # ── 2. LLM + streaming TTS + playback ──
            self._set_state(ConversationState.SPEAKING)

            # Sentence queue: LLM callback detects sentence ends and enqueues
            sentence_queue: queue.Queue[str] = queue.Queue()
            generation_done = threading.Event()
            sentence_buf: list[str] = []
            last_flush_time = [time.time()]

            def _flush_buf():
                """Push buffered text to TTS queue."""
                text = "".join(sentence_buf).strip()
                if text:
                    sentence_queue.put(text)
                    sentence_buf.clear()
                    last_flush_time[0] = time.time()

            def on_token(token: str):
                """Per-token callback from LLM. Detects sentence boundaries."""
                sentence_buf.append(token)

                # Notify GUI of streaming token
                if self.on_response_token:
                    try:
                        self.on_response_token(token)
                    except Exception:
                        pass

                text = "".join(sentence_buf).strip()
                words = len(text.split())

                # 1. Sentence boundary (.!?) — flush full sentences only
                #    This avoids mid-sentence splits that cause audible gaps.
                if text and text[-1] in ".!?" and words >= 2:
                    _flush_buf()
                    return

                # 2. Newline — treat as sentence end
                if text and text[-1] == "\n" and text.strip():
                    _flush_buf()
                    return

            # Stream worker: synthesize sentences and pipe PCM into one
            # continuous aplay process — no gaps between sentences.
            def stream_worker():
                """Synthesize sentences and write PCM to a single audio stream."""
                aplay_open = False

                while self._state == ConversationState.SPEAKING:
                    try:
                        sentence = sentence_queue.get(timeout=0.15)
                    except queue.Empty:
                        if generation_done.is_set() and sentence_queue.empty():
                            break
                        continue

                    if self._state != ConversationState.SPEAKING:
                        break

                    pcm = self._tts.synthesize(sentence)
                    if pcm is None or self._state != ConversationState.SPEAKING:
                        continue

                    # Open the audio stream on the first chunk
                    if not aplay_open:
                        self._audio_output.start_stream(self._tts.output_sample_rate)
                        aplay_open = True

                    self._audio_output.write_stream(pcm)

                # Close the stream — aplay plays remaining buffered audio
                if aplay_open:
                    self._audio_output.finish_stream()

            t_stream = threading.Thread(target=stream_worker, daemon=True)
            t_stream.start()

            # Generate (blocks until complete; on_token fires per token)
            t_llm_start = time.time()
            ai_text = self._llm.generate(
                user_message=user_text,
                history=self._history[-self._max_history * 2 :],
                system_prompt=self._build_system_prompt(),
                stream_callback=on_token,
            )
            t_llm = time.time() - t_llm_start

            # Flush any remaining buffered text
            remaining = "".join(sentence_buf).strip()
            if remaining:
                sentence_queue.put(remaining)

            generation_done.set()
            t_stream.join(timeout=30)

            t_total = time.time() - t_start
            logger.info(
                f"Pipeline: STT={t_stt:.2f}s  LLM={t_llm:.2f}s  Total={t_total:.2f}s"
            )

            if not ai_text:
                return

            # Notify GUI with full response
            if self.on_response:
                try:
                    self.on_response(ai_text)
                except Exception:
                    pass

            # Update conversation history
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": ai_text})
            while len(self._history) > self._max_history * 2:
                self._history.pop(0)

            if self._log_conversations:
                self._save_log(user_text, ai_text)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if self.on_error:
                try:
                    self.on_error(str(e))
                except Exception:
                    pass
        finally:
            self._vad.reset()
            if self._state != ConversationState.IDLE:
                self._set_state(ConversationState.IDLE)

    # ------------------------------------------------------------------
    # Interruption handling
    # ------------------------------------------------------------------

    def _handle_interruption(self):
        logger.info("Interruption detected — stopping.")
        self._audio_output.stop()
        self._llm.cancel()
        self._vad.reset()
        self._set_state(ConversationState.IDLE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        base = self._llm.system_prompt
        addons = {
            "brief": " Reply in exactly 1 short sentence. Be concise.",
            "normal": "",
            "detailed": " Provide thorough, detailed answers.",
        }
        prompt = base + addons.get(self._verbosity, "")
        if self._singlish:
            prompt += (
                " Respond in Singlish (Singapore English). "
                "Use lah, lor, leh, meh, hor, sia at the end of sentences naturally. "
                "Use casual Singlish grammar and expressions like 'can', 'cannot', "
                "'got', 'how come', 'walao', 'shiok', 'bo jio'. Keep it natural and fun."
            )
        return prompt

    def _save_log(self, user_text: str, ai_text: str):
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(
                self._log_directory, f"conversation_{date_str}.json"
            )
            entries = []
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    entries = json.load(f)
            entries.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_text,
                "assistant": ai_text,
            })
            with open(log_file, "w") as f:
                json.dump(entries, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save log: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        """Switch between 'ptt' (push-to-talk) and 'continuous' mode."""
        if mode not in ("ptt", "continuous"):
            return
        self._mode = mode
        self._push_to_talk = (mode == "ptt")
        logger.info(f"Mode: {mode}")
        # If switching to continuous while idle, listening starts automatically
        if mode == "continuous" and self._state == ConversationState.IDLE:
            self._vad.reset()

    def set_push_to_talk(self, enabled: bool):
        self.set_mode("ptt" if enabled else "continuous")

    def push_to_talk_pressed(self):
        """Called when the user presses the talk key (spacebar)."""
        if self._mode != "ptt":
            return

        # If the AI is speaking, interrupt it so the user can talk
        if self._state == ConversationState.SPEAKING:
            logger.info("PTT pressed — interrupting AI speech.")
            self._audio_output.stop()
            self._llm.cancel()
            self._vad.reset()
            self._set_state(ConversationState.LISTENING)
            return

        if self._state == ConversationState.IDLE:
            self._vad.reset()
            self._set_state(ConversationState.LISTENING)

    def push_to_talk_released(self):
        """Called when the user releases the talk key."""
        # VAD handles speech-end detection, nothing to do here.
        pass

    def set_singlish(self, enabled: bool):
        """Toggle Singlish response mode."""
        self._singlish = enabled
        logger.info(f"Singlish: {'ON' if enabled else 'OFF'}")

    @property
    def singlish(self) -> bool:
        return self._singlish

    def set_verbosity(self, level: str):
        if level in ("brief", "normal", "detailed"):
            self._verbosity = level
            logger.info(f"Verbosity: {level}")

    def clear_history(self):
        self._history.clear()
        logger.info("History cleared.")

    def get_history(self) -> list[dict]:
        return list(self._history)

    def stop(self):
        self._running = False
        self._audio_output.stop()
        self._audio_input.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._set_state(ConversationState.IDLE)
        logger.info("Conversation manager stopped.")

    @property
    def state(self) -> str:
        return self._state

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def verbosity(self) -> str:
        return self._verbosity

    @property
    def is_running(self) -> bool:
        return self._running
