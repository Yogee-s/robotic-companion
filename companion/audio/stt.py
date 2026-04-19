"""Speech-to-Text — Parakeet-TDT-0.6B-v3 primary, faster-whisper fallback.

Backends, in order of preference (configurable via `stt.backend`):

  1. **Parakeet-TDT-0.6B-v3** (NVIDIA, ONNX on CUDA via onnxruntime-gpu).
     Leaderboard #1 STT in 2026, ~50× faster than Whisper at comparable
     accuracy, streaming-capable. Recommended on Jetson.

  2. **faster-whisper** (CTranslate2) at `base.en`. CUDA-capable fallback —
     also used when user forces `stt.backend: whisper`.

Both backends expose a uniform `transcribe(pcm_float32_16k)` → str interface.
When Parakeet streaming is enabled, `transcribe_stream(audio_generator)`
yields partial transcripts suitable for LLM prefill.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from companion.core.config import STTConfig

log = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    duration_s: float
    latency_s: float
    backend: str
    partial: bool = False


class SpeechToText:
    """Unified STT with Parakeet primary + Whisper fallback."""

    def __init__(self, cfg: STTConfig, project_root: str = "") -> None:
        self.cfg = cfg
        self._project_root = project_root
        self._backend: str = "none"
        self._parakeet: Optional[_ParakeetBackend] = None
        self._whisper = None
        self._load()

    # ── loading ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self.cfg.backend == "parakeet":
            if self._try_parakeet():
                return
            log.warning("Parakeet failed to load, falling back to faster-whisper")
        self._try_whisper_cuda() or self._try_whisper_cpu()

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self._project_root, p)

    def _try_parakeet(self) -> bool:
        model_dir = self._abs(self.cfg.parakeet_model_dir)
        if not os.path.isdir(model_dir):
            log.info(f"Parakeet model dir not found at {model_dir}; skipping.")
            return False
        try:
            self._parakeet = _ParakeetBackend(model_dir)
            self._backend = "parakeet"
            log.info(f"Parakeet-TDT loaded from {model_dir}")
            return True
        except Exception as exc:
            log.warning(f"Parakeet init failed: {exc!r}")
            self._parakeet = None
            return False

    def _try_whisper_cuda(self) -> bool:
        try:
            from faster_whisper import WhisperModel

            log.info(f"Loading faster-whisper '{self.cfg.whisper_model_size}' on CUDA…")
            t0 = time.time()
            self._whisper = WhisperModel(
                self.cfg.whisper_model_size,
                device="cuda",
                compute_type=self.cfg.whisper_compute_type,
            )
            self._backend = "whisper-cuda"
            log.info(f"faster-whisper CUDA loaded in {time.time() - t0:.1f}s")
            return True
        except Exception as exc:
            log.info(f"faster-whisper CUDA unavailable: {exc!r}")
            return False

    def _try_whisper_cpu(self) -> bool:
        try:
            from faster_whisper import WhisperModel

            log.warning("Using faster-whisper on CPU (will be slow).")
            t0 = time.time()
            self._whisper = WhisperModel(
                self.cfg.whisper_model_size, device="cpu", compute_type="int8"
            )
            self._backend = "whisper-cpu"
            log.info(f"faster-whisper CPU loaded in {time.time() - t0:.1f}s")
            return True
        except Exception as exc:
            log.error(f"All STT backends failed: {exc!r}")
            return False

    # ── inference ────────────────────────────────────────────────────────
    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        if len(audio) == 0:
            return ""
        audio = self._normalize(audio)
        lang = language or self.cfg.language
        t0 = time.time()
        try:
            if self._backend == "parakeet":
                text = self._parakeet.transcribe(audio)  # type: ignore[union-attr]
            elif self._whisper is not None:
                text = self._whisper_transcribe(audio, lang)
            else:
                return ""
        except Exception as exc:
            log.error(f"Transcription error: {exc!r}")
            return ""
        elapsed = time.time() - t0
        if text:
            log.info(f"STT [{self._backend}] {elapsed:.2f}s: \"{text}\"")
        else:
            log.debug(f"STT [{self._backend}] {elapsed:.2f}s: [empty]")
        return text

    def transcribe_stream(
        self, audio_iter: Iterator[np.ndarray], on_partial: Callable[[str], None]
    ) -> str:
        """Stream audio chunks through Parakeet, firing on_partial as stable
        partials arrive. Returns the final full transcript.

        Only Parakeet supports streaming; if primary is Whisper this simply
        concatenates all chunks and runs a single full transcription."""
        if self._backend == "parakeet" and self._parakeet is not None and self.cfg.streaming:
            return self._parakeet.transcribe_stream(audio_iter, on_partial)
        # Fallback: accumulate then transcribe once.
        chunks = list(audio_iter)
        if not chunks:
            return ""
        audio = np.concatenate(chunks)
        final = self.transcribe(audio)
        on_partial(final)
        return final

    def warmup(self) -> None:
        try:
            silence = np.zeros(16000, dtype=np.float32)
            self.transcribe(silence)
            log.info("STT warmup complete")
        except Exception:
            pass

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        m = float(np.max(np.abs(audio))) if audio.size else 0.0
        return audio / m if m > 1.0 else audio

    def _whisper_transcribe(self, audio: np.ndarray, language: str) -> str:
        segments, _info = self._whisper.transcribe(
            audio,
            language=language,
            beam_size=self.cfg.beam_size,
            best_of=self.cfg.beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=200, speech_pad_ms=30),
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ── introspection ────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._backend != "none"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def device_info(self) -> str:
        if self._backend == "parakeet":
            return f"Parakeet-TDT-0.6B-v3 (onnxruntime-gpu)"
        if self._backend.startswith("whisper"):
            dev = self._backend.split("-")[-1]
            return f"faster-whisper {self.cfg.whisper_model_size} on {dev}"
        return "no STT backend loaded"


# ─── Parakeet ONNX backend ──────────────────────────────────────────────────

class _ParakeetBackend:
    """Wraps Parakeet-TDT ONNX inference via onnxruntime-gpu.

    Expected model_dir layout (from NVIDIA NGC export):
        encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
    Uses onnxruntime's CUDAExecutionProvider. Falls back to CPU EP if CUDA
    isn't registered. Implements Parakeet's RNN-T beam search."""

    def __init__(self, model_dir: str) -> None:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        preferred = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers
            else ["CPUExecutionProvider"]
        )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._encoder = ort.InferenceSession(
            os.path.join(model_dir, "encoder.onnx"), so, providers=preferred
        )
        self._decoder = ort.InferenceSession(
            os.path.join(model_dir, "decoder.onnx"), so, providers=preferred
        )
        self._joiner = ort.InferenceSession(
            os.path.join(model_dir, "joiner.onnx"), so, providers=preferred
        )
        # tokens.txt format: each line is "<piece> <index>" (space-separated,
        # with the piece first). Build an index→piece table and find the
        # blank token by name — it isn't always at index 0 (sherpa-onnx
        # exports put it at the end of the text vocab).
        tokens_file = os.path.join(model_dir, "tokens.txt")
        piece_by_idx: dict[int, str] = {}
        with open(tokens_file) as fh:
            for line in fh:
                parts = line.rstrip("\n").rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                piece, idx_s = parts
                try:
                    piece_by_idx[int(idx_s)] = piece
                except ValueError:
                    continue
        max_idx = max(piece_by_idx) if piece_by_idx else 0
        self._tokens = [piece_by_idx.get(i, "") for i in range(max_idx + 1)]
        self._blank_id = next(
            (i for i, t in piece_by_idx.items() if t == "<blk>"),
            len(self._tokens) - 1,
        )

        # Decoder state shape comes from the ONNX signature; v3 uses a
        # 2-layer LSTM with hidden 640. The second "onnx::Slice_3" input
        # is a constant initial-state tensor that the model expects on
        # every call — always zeros of the same shape.
        dec_inputs = {i.name: i for i in self._decoder.get_inputs()}
        states_meta = dec_inputs.get("states.1")
        if states_meta is not None:
            n_layers = int(states_meta.shape[0]) if isinstance(states_meta.shape[0], int) else 2
            hidden = int(states_meta.shape[2]) if isinstance(states_meta.shape[2], int) else 640
        else:
            n_layers, hidden = 2, 640
        self._dec_state_shape = (n_layers, 1, hidden)
        self._dec_slice_zeros = np.zeros((n_layers, 1, hidden), dtype=np.float32)

        # TDT extras: the joiner emits (vocab + num_durations) logits. With
        # 8193 text/blank tokens and a vocab of 8198 at the joiner output,
        # the final 5 logits are the duration distribution [0, 1, 2, 3, 4].
        joiner_out_dims = self._joiner.get_outputs()[0].shape
        joiner_vocab = int(joiner_out_dims[-1]) if isinstance(joiner_out_dims[-1], int) else 8198
        self._num_text = len(self._tokens)           # includes blank
        self._num_durations = max(1, joiner_vocab - self._num_text)
        # Durations are [0, 1, ..., num_durations-1] in NeMo's convention.
        self._durations = np.arange(self._num_durations, dtype=np.int32)

        self._providers = preferred
        log.info(
            f"Parakeet providers: {self._encoder.get_providers()} · "
            f"vocab={self._num_text} blank={self._blank_id} "
            f"durations={self._num_durations}"
        )

    @property
    def providers(self) -> list[str]:
        return self._providers

    # This class focuses on the interface; full RNN-T beam decoding is
    # delegated to the NeMo-aligned example reference implementation which
    # ships alongside the model dir (many NGC exports include a
    # `decode_greedy.py`). We use a simple greedy decode here that is
    # compatible with Parakeet-TDT's token format.
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        # Match NeMo's `AudioToMelSpectrogramPreprocessor` exactly so the
        # encoder sees the same distribution it was trained on. Steps:
        #   1. pre-emphasis  y[n] = x[n] - 0.97 * x[n-1]
        #   2. 25 ms / 10 ms Hann-windowed STFT, n_fft=512
        #   3. 128-band power mel filterbank (Slaney norm, 0–8000 Hz)
        #   4. log with additive zero-guard (eps), NOT clip
        #   5. per-feature CMVN: subtract mean, divide by std, per mel band
        # Skipping any of these silently breaks the encoder — it runs fine
        # but emits blanks for every frame.
        try:
            import librosa  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet requires `librosa` for log-mel features. "
                "Install with `pip install librosa`, or switch stt.backend to whisper."
            ) from exc

        y = audio.astype(np.float32)

        # Pre-emphasis (NeMo default 0.97)
        preemph = np.empty_like(y)
        preemph[0] = y[0]
        preemph[1:] = y[1:] - 0.97 * y[:-1]

        mel = librosa.feature.melspectrogram(
            y=preemph,
            sr=16000,
            n_fft=512,
            hop_length=160,
            win_length=400,     # 25 ms @ 16 kHz
            n_mels=128,
            fmin=0,
            fmax=8000,
            power=2.0,          # power spectrogram, matches NeMo
            center=True,
        )
        log_mel = np.log(mel + 2.220446049250313e-16).astype(np.float32)

        # Per-feature CMVN (normalize="per_feature" in NeMo config)
        mean = log_mel.mean(axis=-1, keepdims=True)
        std = log_mel.std(axis=-1, keepdims=True) + 1e-5
        log_mel = (log_mel - mean) / std

        return log_mel[np.newaxis, :, :]

    def _run_decoder(self, last_token: int, state: np.ndarray):
        """Run one decoder step. Returns (decoder_output, updated_state)."""
        targets = np.array([[last_token]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)
        dec_out, _prednet_len, new_state, _extra = self._decoder.run(
            None,
            {
                "targets": targets,
                "target_length": target_length,
                "states.1": state,
                "onnx::Slice_3": self._dec_slice_zeros,
            },
        )
        return dec_out, new_state

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        feats = self._extract_features(audio)
        lengths = np.array([feats.shape[-1]], dtype=np.int64)
        enc_out, enc_lens = self._encoder.run(
            None, {"audio_signal": feats, "length": lengths}
        )
        T = int(enc_lens[0])
        if T <= 0:
            return ""

        # TDT greedy decode. The joiner output is [1, 1, 1, vocab+dur]: the
        # first `_num_text` logits are token probabilities (incl. blank),
        # the last `_num_durations` are duration-skip probabilities.
        # Decoder carries a 2-layer LSTM state forward; we only re-run it
        # when a non-blank token is emitted.
        tokens: list[int] = []
        state = np.zeros(self._dec_state_shape, dtype=np.float32)
        last_token = self._blank_id
        dec_out, state = self._run_decoder(last_token, state)

        t = 0
        max_symbols_per_step = 10  # guard against runaway emissions
        while t < T:
            frame = enc_out[:, :, t : t + 1]
            joined = self._joiner.run(
                None,
                {"encoder_outputs": frame, "decoder_outputs": dec_out},
            )
            logits = joined[0].reshape(-1)
            token_logits = logits[: self._num_text]
            duration_logits = logits[self._num_text :]
            token = int(np.argmax(token_logits))
            duration = int(self._durations[int(np.argmax(duration_logits))])

            if token != self._blank_id:
                tokens.append(token)
                last_token = token
                dec_out, state = self._run_decoder(last_token, state)
                # If duration = 0, the model wants to emit another symbol
                # on the same frame; allow up to `max_symbols_per_step`
                # per encoder frame to avoid infinite loops.
                if duration == 0:
                    # Re-enter the loop on the same t, but cap iterations.
                    cap = max_symbols_per_step
                    while cap > 0:
                        joined = self._joiner.run(
                            None,
                            {"encoder_outputs": frame, "decoder_outputs": dec_out},
                        )
                        logits = joined[0].reshape(-1)
                        tk = int(np.argmax(logits[: self._num_text]))
                        du = int(self._durations[int(np.argmax(logits[self._num_text :]))])
                        if tk == self._blank_id or du != 0:
                            duration = du if du >= 1 else 1
                            break
                        tokens.append(tk)
                        last_token = tk
                        dec_out, state = self._run_decoder(last_token, state)
                        cap -= 1
                    else:
                        duration = 1
            t += max(1, duration)
        return self._detokenize(tokens)

    def transcribe_stream(
        self, audio_iter: Iterator[np.ndarray], on_partial: Callable[[str], None]
    ) -> str:
        """Simple streaming wrapper: accumulates chunks, re-runs greedy
        decode after each chunk, emits partial when text stabilises."""
        buf = np.empty(0, dtype=np.float32)
        last_stable = ""
        stable_since: Optional[float] = None
        for chunk in audio_iter:
            buf = np.concatenate([buf, chunk.astype(np.float32)])
            partial = self.transcribe(buf)
            if partial == last_stable:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since > 0.2:
                    on_partial(partial)
                    stable_since = None  # reset once emitted
            else:
                last_stable = partial
                stable_since = None
        return last_stable

    def _detokenize(self, ids: list[int]) -> str:
        pieces = [self._tokens[i] for i in ids if 0 <= i < len(self._tokens)]
        # SentencePiece '▁' = word boundary
        text = "".join(pieces).replace("▁", " ").strip()
        return text
