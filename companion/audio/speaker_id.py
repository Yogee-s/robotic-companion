"""Speaker identification — NeMo TitaNet-L embeddings + cosine NN match.

Embeds each user turn to a 192-d vector and compares against a JSON-backed
store of `{name: embedding}`. First-time voices prompt for enrolment.
Used to scope memory (Mem0) per speaker and to personalise replies.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class SpeakerID:
    def __init__(
        self,
        model_path: str,
        speakers_file: str,
        match_threshold: float = 0.65,
    ) -> None:
        self.model_path = model_path
        self.speakers_file = speakers_file
        self.match_threshold = float(match_threshold)
        self._session = None
        self._speakers: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._load()
        self._load_speakers()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            log.warning(
                f"TitaNet model not found at {self.model_path}; speaker ID disabled."
            )
            return
        try:
            import onnxruntime as ort  # type: ignore

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            log.info(f"TitaNet-L loaded ({providers[0]})")
        except Exception as exc:
            log.warning(f"TitaNet load failed: {exc!r}")

    def _load_speakers(self) -> None:
        if not os.path.exists(self.speakers_file):
            return
        try:
            with open(self.speakers_file) as fh:
                raw = json.load(fh)
            self._speakers = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
            log.info(f"Loaded {len(self._speakers)} known speakers")
        except Exception as exc:
            log.warning(f"Failed to load speakers file: {exc!r}")

    def _save_speakers(self) -> None:
        os.makedirs(os.path.dirname(self.speakers_file) or ".", exist_ok=True)
        serialisable = {k: v.tolist() for k, v in self._speakers.items()}
        with open(self.speakers_file, "w") as fh:
            json.dump(serialisable, fh, indent=2)

    @property
    def available(self) -> bool:
        return self._session is not None

    def embed(self, audio_float32_16k: np.ndarray) -> Optional[np.ndarray]:
        if self._session is None or audio_float32_16k.size == 0:
            return None
        try:
            # TitaNet expects log-mel features at 16 kHz; for simplicity
            # we pass the raw waveform and rely on the exported preprocessing
            # head. If the model demands mel input, callers should pre-compute.
            audio = audio_float32_16k.astype(np.float32).reshape(1, -1)
            lengths = np.array([audio.shape[1]], dtype=np.int64)
            out = self._session.run(None, {"audio_signal": audio, "length": lengths})
            emb = out[0].flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as exc:
            log.debug(f"TitaNet inference failed: {exc!r}")
            return None

    def identify(self, audio: np.ndarray) -> tuple[Optional[str], float]:
        """Return (name, similarity) or (None, 0.0) if unknown."""
        emb = self.embed(audio)
        if emb is None or not self._speakers:
            return None, 0.0
        best_name = None
        best_score = -1.0
        for name, ref in self._speakers.items():
            score = float(np.dot(emb, ref))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= self.match_threshold:
            return best_name, best_score
        return None, best_score

    def enrol(self, name: str, audio: np.ndarray) -> bool:
        emb = self.embed(audio)
        if emb is None:
            return False
        with self._lock:
            self._speakers[name] = emb
            self._save_speakers()
        log.info(f"Enrolled speaker '{name}'")
        return True

    def forget(self, name: str) -> bool:
        with self._lock:
            if name not in self._speakers:
                return False
            del self._speakers[name]
            self._save_speakers()
        log.info(f"Removed speaker '{name}'")
        return True

    def known_speakers(self) -> list[str]:
        return list(self._speakers.keys())
