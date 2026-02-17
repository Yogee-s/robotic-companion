"""
Voice Activity Detection using Silero VAD.

Detects speech start/end with a state machine and configurable
silence thresholds.
"""

import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class VADState:
    IDLE = "idle"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"


class VoiceActivityDetector:
    """
    Real-time VAD using Silero (GPU-accelerated via PyTorch).

    State machine: IDLE → SPEECH_START → SPEAKING → SPEECH_END → IDLE
    """

    def __init__(self, config: dict):
        self.threshold = config.get("threshold", 0.5)
        self.silence_duration_ms = config.get("silence_duration_ms", 600)
        self.min_speech_duration_ms = config.get("min_speech_duration_ms", 250)
        self.sample_rate = 16000

        self._state = VADState.IDLE
        self._model = None
        self._torch = None  # Cached torch module reference
        self._speech_buffer: list[np.ndarray] = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._chunk_ms = 32

        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_vad_probability = None

        self._load_model()

    def _load_model(self):
        """Load Silero VAD model (ONNX preferred — no torchaudio needed)."""
        try:
            import torch
            self._torch = torch

            start = time.time()

            # Prefer the silero-vad pip package (ONNX, no torchaudio)
            try:
                from silero_vad import load_silero_vad
                self._model = load_silero_vad(onnx=True)
                elapsed = time.time() - start
                logger.info(f"Silero VAD (ONNX) loaded in {elapsed:.1f}s")
                return
            except ImportError:
                pass

            # Fallback: torch.hub (requires torchaudio)
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            elapsed = time.time() - start
            logger.info(f"Silero VAD loaded in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            logger.warning("Falling back to energy-based VAD.")

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Process one audio chunk through the VAD state machine.

        Args:
            audio_chunk: float32 numpy array at 16 kHz.

        Returns:
            Current state string.
        """
        self._chunk_ms = len(audio_chunk) / self.sample_rate * 1000

        # Get speech probability
        prob = self._get_probability(audio_chunk)

        if self.on_vad_probability:
            try:
                self.on_vad_probability(prob)
            except Exception:
                pass

        is_speech = prob >= self.threshold

        # State machine
        if self._state == VADState.IDLE:
            if is_speech:
                self._state = VADState.SPEECH_START
                self._speech_buffer = [audio_chunk.copy()]
                self._speech_ms = self._chunk_ms
                self._silence_ms = 0
                if self.on_speech_start:
                    try:
                        self.on_speech_start()
                    except Exception:
                        pass

        elif self._state in (VADState.SPEECH_START, VADState.SPEAKING):
            self._speech_buffer.append(audio_chunk.copy())
            self._state = VADState.SPEAKING

            if is_speech:
                self._silence_ms = 0
                self._speech_ms += self._chunk_ms
            else:
                self._silence_ms += self._chunk_ms
                if self._silence_ms >= self.silence_duration_ms:
                    self._state = VADState.SPEECH_END

                    if self._speech_ms >= self.min_speech_duration_ms:
                        full_audio = np.concatenate(self._speech_buffer)
                        if self.on_speech_end:
                            try:
                                self.on_speech_end(full_audio)
                            except Exception as e:
                                logger.error(f"on_speech_end error: {e}")
                    else:
                        logger.debug(
                            f"Speech too short ({self._speech_ms:.0f}ms), ignored."
                        )

                    self._speech_buffer = []
                    self._speech_ms = 0
                    self._silence_ms = 0
                    self._state = VADState.IDLE

        return self._state

    def _get_probability(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability from Silero or energy fallback."""
        if self._model is not None and self._torch is not None:
            try:
                tensor = self._torch.from_numpy(audio_chunk).float()
                return float(self._model(tensor, self.sample_rate).item())
            except Exception:
                pass

        # Energy-based fallback
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return min(1.0, max(0.0, (rms - 0.005) * 10))

    def reset(self):
        self._state = VADState.IDLE
        self._speech_buffer = []
        self._silence_ms = 0
        self._speech_ms = 0
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_speaking(self) -> bool:
        return self._state in (VADState.SPEECH_START, VADState.SPEAKING)
