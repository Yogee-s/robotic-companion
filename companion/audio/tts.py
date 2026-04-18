"""
Text-to-Speech engine.

Supports two engines loaded simultaneously:
  - Piper  (default) — fast, low latency, slightly robotic
  - Kokoro (optional) — natural, human-like, slower (~1.5x real-time)

Both are loaded at init so switching is instant via set_engine().
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

_ESPEAK_LIB_PATHS = [
    "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
    "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
    "/usr/lib/libespeak-ng.so.1",
    "/usr/lib/libespeak-ng.so",
]


class TextToSpeech:
    """
    TTS engine with Piper (fast) and Kokoro (natural).

    Both engines are loaded at init. Switch at runtime with set_engine().
    """

    def __init__(self, config, project_root: str = ""):
        # Accept either a legacy dict or the new TTSConfig dataclass.
        get = (lambda k, d=None: getattr(config, k, d)) if not isinstance(config, dict) else (lambda k, d=None: config.get(k, d))
        self._default_engine = get("engine", "kokoro")
        self.voice = get("voice", "af_heart")
        self.speaking_rate = get("speaking_rate", 1.1)
        self.output_sample_rate = get("output_sample_rate", 24000)
        self.lang = get("lang", "en-us")
        self._piper_model = get("piper_model", "en_US-hfc_female-medium")
        self._loud_fallback = get("loud_fallback", True)

        if not project_root:
            # tts.py lives at companion/audio/tts.py — go up two directories.
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        self._models_dir = os.path.join(project_root, "models")

        # Engine instances (both loaded if available)
        self._kokoro = None
        self._kokoro_sr = 24000
        self._piper_voice = None
        self._piper_sr = 22050
        self._piper_python_api = False
        self._piper_cli = False

        self._active_engine = None
        self._available = False
        self._has_kokoro = False
        self._has_piper = False

        self._load_all()

    # ------------------------------------------------------------------
    # Load both engines at startup
    # ------------------------------------------------------------------

    def _load_all(self):
        """Lazy-load: only instantiate the configured primary engine at init.
        The alternate engine is loaded on demand (e.g. from `set_engine()`)."""
        if self._default_engine == "kokoro":
            self._has_kokoro = self._load_kokoro()
            if self._has_kokoro:
                self._activate("kokoro")
                return
            if self._loud_fallback:
                logger.error(
                    "Kokoro TTS failed to load — falling back to Piper. "
                    "Check models/kokoro/ and `pip install kokoro-onnx`."
                )
            self._has_piper = self._load_piper()
            if self._has_piper:
                self._activate("piper")
        else:
            self._has_piper = self._load_piper()
            if self._has_piper:
                self._activate("piper")
                return
            if self._loud_fallback:
                logger.error("Piper TTS failed — falling back to Kokoro.")
            self._has_kokoro = self._load_kokoro()
            if self._has_kokoro:
                self._activate("kokoro")
        if not self._available:
            logger.error("No TTS engine available!")

    def _activate(self, engine: str):
        """Switch the active engine."""
        if engine == "kokoro" and self._has_kokoro:
            self._active_engine = "kokoro"
            self.output_sample_rate = self._kokoro_sr
            self._available = True
            logger.info(f"TTS active: Kokoro ({self.voice})")
        elif self._has_piper:
            self._active_engine = "piper"
            self.output_sample_rate = self._piper_sr
            self._available = True
            logger.info(f"TTS active: Piper ({self._piper_model})")
        elif self._piper_cli:
            self._active_engine = "piper-cli"
            self._available = True

    def _load_kokoro(self) -> bool:
        """Load Kokoro TTS (ONNX FP16) — natural voice with optimized CPU session."""
        try:
            from kokoro_onnx import Kokoro

            kokoro_dir = os.path.join(self._models_dir, "kokoro")
            voices_path = os.path.join(kokoro_dir, "voices-v1.0.bin")

            # Prefer FP16 (faster on ARM), fall back to FP32
            fp16_path = os.path.join(kokoro_dir, "kokoro-v1.0.fp16.onnx")
            fp32_path = os.path.join(kokoro_dir, "kokoro-v1.0.onnx")
            model_path = fp16_path if os.path.exists(fp16_path) else fp32_path

            if not os.path.exists(model_path) or not os.path.exists(voices_path):
                logger.info("Kokoro model files not found — skipping.")
                return False

            start = time.time()
            espeak_kwargs = self._get_espeak_config()

            # Build optimized CPU session (CUDA is slower for this model)
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = os.cpu_count() or 6
            sess_opts.inter_op_num_threads = 2
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )

            self._kokoro = Kokoro.from_session(sess, voices_path, **espeak_kwargs)

            test_samples, test_sr = self._kokoro.create(
                "test", voice=self.voice, speed=1.0, lang=self.lang
            )
            if test_samples is None or len(test_samples) == 0:
                self._kokoro = None
                return False

            self._kokoro_sr = test_sr
            variant = "fp16" if "fp16" in model_path else "fp32"
            logger.info(
                f"Kokoro TTS loaded in {time.time() - start:.1f}s "
                f"({variant}, CPU, {sess_opts.intra_op_num_threads} threads)"
            )
            return True
        except ImportError:
            logger.info("kokoro-onnx not installed — Kokoro unavailable.")
            return False
        except Exception as e:
            logger.warning(f"Kokoro load failed: {e}")
            self._kokoro = None
            return False

    @staticmethod
    def _get_espeak_config() -> dict:
        try:
            from kokoro_onnx import EspeakConfig
            for path in _ESPEAK_LIB_PATHS:
                if os.path.exists(path):
                    return {"espeak_config": EspeakConfig(lib_path=path)}
        except ImportError:
            pass
        return {}

    def _load_piper(self) -> bool:
        """Load Piper TTS — fast voice."""
        piper_dir = os.path.join(self._models_dir, "piper")
        model_path = os.path.join(piper_dir, f"{self._piper_model}.onnx")
        config_path = os.path.join(piper_dir, f"{self._piper_model}.onnx.json")

        try:
            from piper import PiperVoice
            if not os.path.exists(model_path):
                logger.warning(f"Piper model not found: {model_path}")
                return False

            start = time.time()
            self._piper_voice = PiperVoice.load(model_path, config_path=config_path)
            try:
                self._piper_sr = self._piper_voice.config.sample_rate
            except Exception:
                pass

            self._piper_python_api = self._test_piper_api()
            logger.info(
                f"Piper TTS loaded in {time.time() - start:.1f}s "
                f"(python_api={'OK' if self._piper_python_api else 'CLI'})"
            )
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Piper load failed: {e}")

        # Try CLI
        try:
            result = subprocess.run(
                ["piper", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self._piper_cli = True
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def _test_piper_api(self) -> bool:
        try:
            for chunk in self._piper_voice.synthesize("test"):
                if chunk is not None:
                    pcm = self._chunk_to_pcm(chunk)
                    if pcm and len(pcm) > 0:
                        return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_engine(self, engine: str):
        """Switch active engine at runtime: 'piper' or 'kokoro'.
        Lazy-loads the alternate engine on first use."""
        if engine == "kokoro":
            if not self._has_kokoro:
                self._has_kokoro = self._load_kokoro()
            if self._has_kokoro:
                self._activate("kokoro")
                return
        elif engine == "piper":
            if not self._has_piper:
                self._has_piper = self._load_piper()
            if self._has_piper:
                self._activate("piper")
                return
        logger.warning(f"TTS engine '{engine}' not available.")

    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to raw PCM int16 bytes at output_sample_rate."""
        if not text or not text.strip() or not self._available:
            return None

        text = text.strip()
        start = time.time()

        try:
            pcm = None

            if self._active_engine == "kokoro":
                pcm = self._synthesize_kokoro(text)
                if pcm is None and self._loud_fallback:
                    logger.error(
                        "Kokoro synthesis returned no audio — falling back to Piper for this utterance."
                    )

            if pcm is None and self._piper_voice and self._piper_python_api:
                pcm = self._synthesize_piper(text)

            if pcm is None and (self._piper_cli or self._piper_voice):
                pcm = self._synthesize_piper_cli(text)

            if pcm:
                elapsed = time.time() - start
                duration = len(pcm) / 2 / self.output_sample_rate
                logger.info(
                    f"TTS ({elapsed:.2f}s): {duration:.1f}s audio "
                    f"for {len(text)} chars [{self._active_engine}]"
                )
            return pcm

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None

    # ------------------------------------------------------------------
    # Kokoro synthesis
    # ------------------------------------------------------------------

    def _synthesize_kokoro(self, text: str) -> Optional[bytes]:
        try:
            samples, sample_rate = self._kokoro.create(
                text, voice=self.voice, speed=self.speaking_rate, lang=self.lang,
            )
            if samples is None or len(samples) == 0:
                return None
            self.output_sample_rate = sample_rate
            return (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        except Exception as e:
            logger.warning(f"Kokoro synthesis failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Piper synthesis
    # ------------------------------------------------------------------

    def _synthesize_piper(self, text: str) -> Optional[bytes]:
        sr = self.output_sample_rate
        syn_config = None
        try:
            from piper.config import SynthesisConfig
            length_scale = 1.0 / self.speaking_rate if self.speaking_rate > 0 else 1.0
            syn_config = SynthesisConfig(length_scale=length_scale)
        except ImportError:
            pass

        try:
            chunks = []
            kwargs = {"syn_config": syn_config} if syn_config else {}
            for chunk in self._piper_voice.synthesize(text, **kwargs):
                if hasattr(chunk, "sample_rate") and chunk.sample_rate:
                    sr = chunk.sample_rate
                pcm = self._chunk_to_pcm(chunk)
                if pcm:
                    chunks.append(pcm)
            if chunks:
                self.output_sample_rate = sr
                return b"".join(chunks)
        except Exception as e:
            logger.debug(f"Piper synthesize() failed: {e}")

        if hasattr(self._piper_voice, "synthesize_stream_raw"):
            try:
                chunks = list(self._piper_voice.synthesize_stream_raw(text))
                if chunks:
                    return b"".join(chunks)
            except Exception:
                pass

        if hasattr(self._piper_voice, "synthesize_wav"):
            try:
                wav_bytes = self._piper_voice.synthesize_wav(text)
                if wav_bytes and len(wav_bytes) > 44:
                    buf = io.BytesIO(wav_bytes)
                    with wave.open(buf, "rb") as rf:
                        self.output_sample_rate = rf.getframerate()
                        return rf.readframes(rf.getnframes())
            except Exception:
                pass
        return None

    @staticmethod
    def _chunk_to_pcm(chunk) -> Optional[bytes]:
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

    def _synthesize_piper_cli(self, text: str) -> Optional[bytes]:
        piper_dir = os.path.join(self._models_dir, "piper")
        model_path = os.path.join(piper_dir, f"{self._piper_model}.onnx")
        model_arg = model_path if os.path.exists(model_path) else self._piper_model
        rate_args = []
        if self.speaking_rate != 1.0:
            rate_args = ["--length-scale", str(1.0 / self.speaking_rate)]

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
            return None
        except Exception:
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def active_engine(self) -> str:
        return self._active_engine or "none"

    @property
    def available_engines(self) -> list[str]:
        engines = []
        if self._has_piper:
            engines.append("piper")
        if self._has_kokoro:
            engines.append("kokoro")
        return engines

    @property
    def info(self) -> str:
        if self._active_engine == "kokoro":
            return f"Kokoro ({self.voice}) — natural voice"
        if self._active_engine in ("piper", "piper-cli"):
            return f"Piper ({self._piper_model}) — fast voice"
        return "TTS: not available"
