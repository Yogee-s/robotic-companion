"""Moondream-2 VLM sidecar.

Loaded via llama-cpp-python's multimodal path (mmproj). Gives the robot
eyes: scene captioning + visual question answering. Always runs on its
own cadence (1-2 Hz in a background thread) so it never blocks the
conversation loop; per-query latency on Jetson Orin Nano is ~400-700 ms.

Two entry points:
  - `caption(frame)` → one-line scene description (for ambient awareness)
  - `answer(frame, question)` → VQA for "what is this?" style questions
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class MoondreamVLM:
    def __init__(
        self,
        model_path: str,
        mmproj_path: str = "",
        enabled: bool = True,
        max_tokens: int = 80,
    ) -> None:
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.enabled = enabled
        self.max_tokens = int(max_tokens)
        self._llm = None
        self._handler = None
        self._lock = threading.Lock()
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            log.warning(f"Moondream model not found at {self.model_path}; VLM disabled.")
            self.enabled = False
            return
        try:
            from llama_cpp import Llama  # type: ignore
            from llama_cpp.llama_chat_format import MoondreamChatHandler  # type: ignore
        except ImportError:
            log.warning("llama-cpp-python lacks multimodal support — VLM disabled.")
            self.enabled = False
            return
        try:
            handler = MoondreamChatHandler(clip_model_path=self.mmproj_path)
            self._llm = Llama(
                model_path=self.model_path,
                chat_handler=handler,
                n_gpu_layers=-1,
                n_ctx=2048,
                n_batch=256,
                verbose=False,
            )
            self._handler = handler
            log.info(f"Moondream-2 VLM loaded (model={self.model_path})")
        except Exception as exc:
            log.warning(f"Moondream load failed: {exc!r}")
            self.enabled = False
            self._llm = None

    @property
    def available(self) -> bool:
        return self.enabled and self._llm is not None

    # ── public API ───────────────────────────────────────────────────────
    def caption(self, frame_bgr: np.ndarray) -> Optional[str]:
        return self._ask(frame_bgr, "Describe the scene in one sentence.")

    def answer(self, frame_bgr: np.ndarray, question: str) -> Optional[str]:
        return self._ask(frame_bgr, question)

    # ── implementation ───────────────────────────────────────────────────
    def _ask(self, frame_bgr: np.ndarray, question: str) -> Optional[str]:
        if not self.available or frame_bgr is None or frame_bgr.size == 0:
            return None
        image_uri = self._encode(frame_bgr)
        with self._lock:
            try:
                assert self._llm is not None
                out = self._llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_uri}},
                                {"type": "text", "text": question},
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                )
                return out["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                log.debug(f"Moondream inference failed: {exc!r}")
                return None

    @staticmethod
    def _encode(frame_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
