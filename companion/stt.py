"""
Speech-to-Text (STT) using faster-whisper or openai-whisper.

Tries the fastest available backend:
  1. faster-whisper on CUDA (CTranslate2 — if compiled with CUDA)
  2. openai-whisper on CUDA (PyTorch — if torch.cuda is available)
  3. faster-whisper on CPU (fallback)
"""

import logging
import time
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Speech-to-Text engine.

    Automatically picks the best backend (CUDA preferred).
    """

    def __init__(self, config: dict):
        self.model_size = config.get("model_size", "base.en")
        self.device = config.get("device", "cuda")
        self.compute_type = config.get("compute_type", "int8")
        self.beam_size = config.get("beam_size", 5)
        self.language = config.get("language", "en")

        self._model = None
        self._backend = None  # "faster-whisper" or "openai-whisper"
        self._load_model()

    def _load_model(self):
        """Try backends in order of preference."""
        # 1. faster-whisper on CUDA
        if self.device == "cuda":
            if self._try_faster_whisper_cuda():
                return

        # 2. openai-whisper on CUDA (PyTorch)
        if self.device == "cuda":
            if self._try_openai_whisper_cuda():
                return

        # 3. faster-whisper on CPU
        self._try_faster_whisper_cpu()

    def _try_faster_whisper_cuda(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            logger.info(
                f"Loading Whisper model '{self.model_size}' "
                f"(device=cuda, compute={self.compute_type})..."
            )
            start = time.time()
            self._model = WhisperModel(
                self.model_size, device="cuda", compute_type=self.compute_type,
            )
            elapsed = time.time() - start
            self.device = "cuda"
            self._backend = "faster-whisper"
            logger.info(f"faster-whisper loaded on CUDA in {elapsed:.1f}s.")
            return True
        except Exception as e:
            logger.info(f"faster-whisper CUDA unavailable: {e}")
            return False

    def _try_openai_whisper_cuda(self) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("PyTorch CUDA not available, skipping openai-whisper CUDA.")
                return False

            import whisper
            logger.info(
                f"Loading openai-whisper '{self.model_size}' on CUDA (PyTorch)..."
            )
            start = time.time()
            self._model = whisper.load_model(self.model_size, device="cuda")
            elapsed = time.time() - start
            self.device = "cuda"
            self.compute_type = "fp16"
            self._backend = "openai-whisper"
            logger.info(f"openai-whisper loaded on CUDA in {elapsed:.1f}s.")
            return True
        except ImportError:
            logger.info("openai-whisper not installed. pip install openai-whisper")
            return False
        except Exception as e:
            logger.warning(f"openai-whisper CUDA failed: {e}")
            return False

    def _try_faster_whisper_cpu(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            logger.warning("Falling back to faster-whisper on CPU.")
            start = time.time()
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8",
            )
            elapsed = time.time() - start
            self.device = "cpu"
            self.compute_type = "int8"
            self._backend = "faster-whisper"
            logger.info(f"faster-whisper loaded on CPU in {elapsed:.1f}s.")
            return True
        except Exception as e:
            logger.error(f"All STT backends failed: {e}")
            self._model = None
            return False

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Float32 numpy array at 16kHz
            language: Language code override (None = use self.language)

        Returns:
            Transcribed text string.
        """
        if self._model is None:
            logger.error("Whisper model not loaded. Cannot transcribe.")
            return ""

        if len(audio) == 0:
            return ""

        lang = language or self.language

        # Ensure float32 normalized to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        try:
            start = time.time()

            if self._backend == "openai-whisper":
                result = self._transcribe_openai(audio, lang)
            else:
                result = self._transcribe_faster(audio, lang)

            elapsed = time.time() - start

            if result:
                logger.info(f"STT ({elapsed:.2f}s): \"{result}\"")
            else:
                logger.debug(f"STT ({elapsed:.2f}s): [no speech detected]")

            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _transcribe_faster(self, audio: np.ndarray, language: str) -> str:
        """Transcribe using faster-whisper (CTranslate2)."""
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            best_of=self.beam_size,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=200,
                speech_pad_ms=30,
            ),
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        parts = [seg.text.strip() for seg in segments]
        return " ".join(parts).strip()

    def _transcribe_openai(self, audio: np.ndarray, language: str) -> str:
        """Transcribe using openai-whisper (PyTorch)."""
        result = self._model.transcribe(
            audio,
            language=language,
            fp16=(self.device == "cuda"),
            beam_size=self.beam_size,
            best_of=self.beam_size,
            temperature=0,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        return result.get("text", "").strip()

    def warmup(self):
        """Run a dummy transcription to JIT-compile / warm GPU caches."""
        if self._model is None:
            return
        try:
            start = time.time()
            dummy = np.zeros(16000, dtype=np.float32)  # 1s silence
            self.transcribe(dummy)
            logger.info(f"STT warmup done ({time.time() - start:.1f}s)")
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_info(self) -> str:
        backend = self._backend or "none"
        return f"Whisper {self.model_size} on {self.device} ({self.compute_type}, {backend})"
