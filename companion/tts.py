"""
Text-to-Speech using Piper TTS.

Fast offline neural TTS. Voice model is loaded ONCE and reused.
"""

import io
import logging
import os
import subprocess
import tempfile
import time
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Piper TTS engine.

    The ONNX voice model is loaded once at init and kept in memory
    for near-instant synthesis on subsequent calls.
    """

    def __init__(self, config: dict):
        self.model_name = config.get("model", "en_US-amy-medium")
        self.speaking_rate = config.get("speaking_rate", 1.0)
        self.output_sample_rate = config.get("output_sample_rate", 22050)

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_dir, "models", "piper")
        self._model_path = os.path.join(model_dir, f"{self.model_name}.onnx")
        self._config_path = os.path.join(model_dir, f"{self.model_name}.onnx.json")

        self._voice = None
        self._use_cli = False
        self._python_api_works = False
        self._available = False

        self._load_voice()

    def _load_voice(self):
        """Load Piper voice model once and cache it."""
        # Try the Python module first (faster, no subprocess)
        try:
            from piper import PiperVoice

            if not os.path.exists(self._model_path):
                logger.warning(f"Piper model not found: {self._model_path}")
            else:
                start = time.time()
                self._voice = PiperVoice.load(
                    self._model_path, config_path=self._config_path
                )
                elapsed = time.time() - start

                # Read actual sample rate from model config
                try:
                    self.output_sample_rate = self._voice.config.sample_rate
                except Exception:
                    pass

                # Quick test: does the Python API actually produce audio?
                self._python_api_works = self._test_python_api()

                self._available = True
                logger.info(
                    f"Piper voice loaded in {elapsed:.1f}s ({self.model_name}), "
                    f"python_api={'OK' if self._python_api_works else 'FAIL (using CLI)'}"
                )
                return
        except ImportError:
            logger.debug("piper Python module not available, trying CLI")
        except Exception as e:
            logger.warning(f"Failed to load Piper voice: {e}")

        # Fallback: Piper CLI
        self._setup_cli()

    def _test_python_api(self) -> bool:
        """Quick test to see if the Python API produces audio."""
        try:
            for chunk in self._voice.synthesize("test"):
                if chunk is not None:
                    pcm = self._chunk_to_pcm(chunk)
                    if pcm and len(pcm) > 0:
                        return True
            return False
        except Exception:
            return False

    def _setup_cli(self):
        """Check if piper CLI is available."""
        try:
            result = subprocess.run(
                ["piper", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self._use_cli = True
                self._available = True
                logger.info(f"Piper CLI available: {result.stdout.strip()}")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        logger.warning("Piper TTS not available. Install: pip install piper-tts")

    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to raw PCM int16 bytes at output_sample_rate.

        Returns PCM bytes or None on error.
        """
        if not text or not text.strip() or not self._available:
            return None

        text = text.strip()
        start = time.time()

        try:
            pcm = None

            # Use Python API if it produces audio; otherwise CLI
            if self._voice is not None and self._python_api_works:
                pcm = self._synthesize_python(text)

            # Fall back to CLI (raw pipe — no temp files)
            if pcm is None:
                pcm = self._synthesize_cli(text)

            if pcm:
                elapsed = time.time() - start
                duration = len(pcm) / 2 / self.output_sample_rate
                logger.info(
                    f"TTS ({elapsed:.2f}s): {duration:.1f}s audio for {len(text)} chars"
                )
            return pcm

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None

    def _synthesize_python(self, text: str) -> Optional[bytes]:
        """Synthesize using the cached PiperVoice instance."""
        sr = self.output_sample_rate

        # Build SynthesisConfig with speaking rate
        syn_config = None
        try:
            from piper.config import SynthesisConfig
            length_scale = 1.0 / self.speaking_rate if self.speaking_rate > 0 else 1.0
            syn_config = SynthesisConfig(length_scale=length_scale)
        except ImportError:
            pass

        # Method 1: synthesize() generator — yields AudioChunk, bytes, or numpy
        try:
            chunks = []
            kwargs = {"syn_config": syn_config} if syn_config else {}
            for chunk in self._voice.synthesize(text, **kwargs):
                # Read sample_rate from AudioChunk if available
                if hasattr(chunk, "sample_rate") and chunk.sample_rate:
                    sr = chunk.sample_rate
                pcm = self._chunk_to_pcm(chunk)
                if pcm:
                    chunks.append(pcm)
            if chunks:
                self.output_sample_rate = sr
                return b"".join(chunks)
        except Exception as e:
            logger.debug(f"synthesize() generator failed: {e}")

        # Method 2: synthesize_stream_raw — yields raw PCM chunks
        if hasattr(self._voice, "synthesize_stream_raw"):
            try:
                chunks = list(self._voice.synthesize_stream_raw(text))
                if chunks:
                    self.output_sample_rate = sr
                    return b"".join(chunks)
            except Exception:
                pass

        # Method 3: synthesize_wav — returns complete WAV bytes
        if hasattr(self._voice, "synthesize_wav"):
            try:
                wav_bytes = self._voice.synthesize_wav(text)
                if wav_bytes and len(wav_bytes) > 44:
                    buf = io.BytesIO(wav_bytes)
                    with wave.open(buf, "rb") as rf:
                        self.output_sample_rate = rf.getframerate()
                        return rf.readframes(rf.getnframes())
            except Exception:
                pass

        logger.debug("All Python TTS methods failed")
        return None

    @staticmethod
    def _chunk_to_pcm(chunk) -> Optional[bytes]:
        """Convert a chunk from the synthesize generator to raw PCM int16 bytes."""
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
            return TextToSpeech._chunk_to_pcm(chunk[0])

        return None

    def _synthesize_cli(self, text: str) -> Optional[bytes]:
        """Synthesize using Piper CLI with raw PCM pipe (no temp files)."""
        model_arg = self._model_path if os.path.exists(self._model_path) else self.model_name
        rate_args = []
        if self.speaking_rate != 1.0:
            rate_args = ["--length-scale", str(1.0 / self.speaking_rate)]

        # Try raw output first (faster — no WAV encoding/decoding)
        try:
            cmd = ["piper", "--model", model_arg, "--output-raw"] + rate_args
            proc = subprocess.run(
                cmd, input=text, capture_output=True, text=True, timeout=30
            )
            if proc.returncode == 0 and proc.stdout:
                # --output-raw gives raw int16 PCM on stdout
                # but subprocess.run with text=True encodes it...
                pass
        except Exception:
            pass

        # Reliable path: output to temp WAV file
        tmp_path = None
        try:
            tmp_path = tempfile.mktemp(suffix=".wav")
            cmd = ["piper", "--model", model_arg, "--output_file", tmp_path] + rate_args
            proc = subprocess.run(
                cmd, input=text, capture_output=True, text=True, timeout=30
            )
            if proc.returncode == 0 and os.path.exists(tmp_path):
                with wave.open(tmp_path, "rb") as wf:
                    self.output_sample_rate = wf.getframerate()
                    return wf.readframes(wf.getnframes())
            if proc.stderr:
                logger.error(f"Piper CLI error: {proc.stderr.strip().split(chr(10))[-1]}")
            return None

        except Exception as e:
            logger.debug(f"CLI synthesis failed: {e}")
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def info(self) -> str:
        if self._voice is not None and self._python_api_works:
            return f"Piper ({self.model_name}) — Python API (in-memory)"
        if self._voice is not None or self._use_cli:
            return f"Piper ({self.model_name}) — CLI fallback"
        return "TTS: not available"
