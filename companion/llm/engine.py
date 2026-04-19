"""
LLM Inference Engine using llama-cpp-python.

Runs quantized LLMs locally with CUDA acceleration on Jetson Orin Nano.
When configured with an mmproj file, the same engine also answers
questions about images (multimodal) — no separate VLM model needed.
"""

import base64
import logging
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Local LLM inference using llama-cpp-python.

    Supports Llama 3.2, Phi-3, Mistral, and other GGUF models.
    """

    def __init__(self, config, model_path: str = "", mmproj_path: str = ""):
        """Accepts either a dict (legacy) or an LLMConfig dataclass.
        `model_path` overrides the config lookup — pass the absolute path
        resolved via `AppConfig.llm_model_path()` when using the new API.
        `mmproj_path` enables multimodal mode (image captioning)."""
        get = (lambda k, d=None: getattr(config, k, d)) if not isinstance(config, dict) else (lambda k, d=None: config.get(k, d))
        self.model_path = model_path or get("model_path", "")
        self.mmproj_path = mmproj_path or get("mmproj_path", "")
        self.n_gpu_layers = get("n_gpu_layers", -1)
        self.context_length = get("context_length", 2048)
        self.max_tokens = get("max_tokens", 120)
        self.temperature = get("temperature", 0.7)
        self.top_p = get("top_p", 0.9)
        self.repeat_penalty = get("repeat_penalty", 1.1)
        self.system_prompt = get(
            "system_prompt",
            "You are a warm, curious AI companion. Keep replies short and finish every sentence."
        )

        self._model = None
        self._lock = threading.Lock()
        self._generating = False
        self._multimodal = False

    def load(self):
        """Load the LLM model, with vision if `mmproj_path` is set."""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading LLM model: {self.model_path}")
            logger.info(f"  GPU layers: {self.n_gpu_layers}, context: {self.context_length}")

            # Try to attach the vision projector if one was configured.
            # Falls back to text-only if the handler can't be created —
            # e.g. llama-cpp-python version without multimodal support.
            chat_handler = None
            if self.mmproj_path and os.path.exists(self.mmproj_path):
                chat_handler = self._make_vision_handler(self.mmproj_path)
                if chat_handler is not None:
                    logger.info(f"  Multimodal: {os.path.basename(self.mmproj_path)}")

            start = time.time()
            kwargs = dict(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.context_length,
                n_batch=512,
                verbose=False,
            )
            if chat_handler is not None:
                kwargs["chat_handler"] = chat_handler
            self._model = Llama(**kwargs)
            self._multimodal = chat_handler is not None
            elapsed = time.time() - start
            logger.info(
                f"LLM model loaded in {elapsed:.1f}s "
                f"({'multimodal' if self._multimodal else 'text-only'})."
            )

        except ImportError:
            logger.error(
                "llama-cpp-python not installed. "
                "Build with CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")

    @staticmethod
    def _make_vision_handler(mmproj_path: str):
        """Pick the right chat handler class for the mmproj we're loading.

        llama-cpp-python ships a handful: MoondreamChatHandler,
        Llava15ChatHandler, Gemma3ChatHandler (newer builds), etc. We try
        the Gemma-specific one first, then fall back to the generic
        Llava15 loader which works for most CLIP-style projectors.

        `verbose=False` is passed where the handler accepts it — silences
        the ~1400-line tensor dump from the CLIP/audio encoder load.
        """
        def _try(cls_name: str):
            try:
                import llama_cpp.llama_chat_format as fmt
                cls = getattr(fmt, cls_name, None)
                if cls is None:
                    return None
                # Newer handlers take verbose=; older ones don't.
                try:
                    return cls(clip_model_path=mmproj_path, verbose=False)
                except TypeError:
                    return cls(clip_model_path=mmproj_path)
            except Exception:
                return None

        for name in ("Gemma3ChatHandler", "Llava15ChatHandler"):
            h = _try(name)
            if h is not None:
                logger.info(f"Multimodal handler: {name}")
                return h
        logger.warning(
            "No compatible multimodal handler in llama-cpp-python. "
            "Loading text-only."
        )
        return None

    @property
    def is_multimodal(self) -> bool:
        """True if the model was loaded with a vision projector."""
        return self._model is not None and self._multimodal

    def answer(self, frame_bgr: "np.ndarray", question: str,
               max_tokens: int = 120) -> Optional[str]:
        """Thin alias: answer a VQA-style user question about a frame.

        Distinguishes the call-site intent (user-directed Q&A) from the
        ambient scene captioning done by `SceneWatcher`. Shares the same
        multimodal plumbing.
        """
        return self.caption(frame_bgr, question, max_tokens=max_tokens)

    def caption(self, frame_bgr: "np.ndarray", question: str,
                max_tokens: int = 80) -> Optional[str]:
        """Answer a question about a BGR image using the loaded model.
        Returns None if the model is not multimodal / not loaded."""
        if not self.is_multimodal:
            return None
        try:
            import cv2
            ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        except Exception as exc:
            logger.debug(f"caption: JPEG encode failed: {exc!r}")
            return None
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": question},
                ],
            },
        ]
        try:
            with self._lock:
                resp = self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.4,   # lower temp for grounded descriptions
                    top_p=0.9,
                )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning(f"caption failed: {exc!r}")
            return None

    def generate(
        self,
        user_message: str,
        history: list[dict] | None = None,
        system_prompt: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            user_message: The user's input text.
            history: List of {"role": "user"|"assistant", "content": str} dicts.
            system_prompt: Override system prompt (None = use default).
            stream_callback: Called with each token as it's generated.

        Returns:
            Complete response text.
        """
        if self._model is None:
            logger.error("LLM model not loaded. Call load() first.")
            return "I'm sorry, my language model isn't loaded yet."

        # Build message list for chat completion
        messages = []

        # System prompt
        sys_prompt = system_prompt or self.system_prompt
        messages.append({"role": "system", "content": sys_prompt})

        # Conversation history
        if history:
            for entry in history:
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"],
                })

        # Current user message
        messages.append({"role": "user", "content": user_message})

        try:
            with self._lock:
                self._generating = True
                start = time.time()

                if stream_callback:
                    return self._generate_streaming(messages, stream_callback, start)
                else:
                    return self._generate_blocking(messages, start)

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I encountered an error while thinking. Could you try again?"
        finally:
            self._generating = False

    def _generate_blocking(self, messages: list[dict], start_time: float) -> str:
        """Non-streaming generation."""
        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
        )

        text = response["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - start_time
        tokens = response.get("usage", {}).get("completion_tokens", len(text.split()))

        logger.info(
            f"LLM ({elapsed:.2f}s, ~{tokens} tokens, "
            f"~{tokens / elapsed:.1f} tok/s): \"{text[:80]}...\""
        )
        return text

    def _generate_streaming(
        self,
        messages: list[dict],
        callback: Callable[[str], None],
        start_time: float,
    ) -> str:
        """Streaming generation with per-token callback."""
        stream = self._model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stream=True,
        )

        full_text = []
        token_count = 0

        for chunk in stream:
            if not self._generating:
                break  # Cancelled (e.g., user interrupted)

            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_text.append(token)
                token_count += 1
                try:
                    callback(token)
                except Exception:
                    pass

        result = self._trim_to_sentence("".join(full_text).strip())
        elapsed = time.time() - start_time

        logger.info(
            f"LLM streamed ({elapsed:.2f}s, {token_count} tokens, "
            f"~{token_count / max(elapsed, 0.01):.1f} tok/s)"
        )
        return result

    @staticmethod
    def _trim_to_sentence(text: str) -> str:
        """Trim text to the last complete sentence so nothing is cut off."""
        if not text:
            return text
        # Already ends cleanly
        if text[-1] in ".!?\"'":
            return text
        # Find the last sentence-ending punctuation
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                return text[: i + 1]
        # No sentence boundary found — return as-is
        return text

    def cancel(self):
        """Cancel an ongoing generation."""
        self._generating = False

    def prefill(
        self,
        user_message: str,
        history: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Warm the KV-cache with a partial user message during STT tail.

        Runs a single-token generation so llama-cpp-python evaluates the
        full prefix (system + history + partial user) and caches it.
        When the real `generate()` call lands with the final transcript a
        few hundred ms later, llama-cpp's session cache replays the
        matching prefix and skips prompt-eval — saving 200-600 ms on
        Jetson where prompt-eval dominates first-token latency.

        Safe to call at most once per turn. Subsequent prefill calls do
        nothing if a real generation is already in flight.
        """
        if self._model is None or self._generating:
            return
        messages: list[dict] = []
        messages.append({"role": "system", "content": system_prompt or self.system_prompt})
        if history:
            for entry in history:
                messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_message})
        try:
            with self._lock:
                # Acquire-and-release pattern: we don't want to hold the
                # lock during the real `generate()` call that follows.
                self._model.create_chat_completion(
                    messages=messages, max_tokens=1, temperature=0.0
                )
        except Exception as exc:
            logger.debug("prefill skipped: %r", exc)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_generating(self) -> bool:
        return self._generating

    @property
    def info(self) -> str:
        if self._model:
            return (
                f"LLM: {self.model_path.split('/')[-1]} | "
                f"GPU layers: {self.n_gpu_layers} | "
                f"Context: {self.context_length}"
            )
        return "LLM: Not loaded"
