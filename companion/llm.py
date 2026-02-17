"""
LLM Inference Engine using llama-cpp-python.

Runs quantized LLMs locally with CUDA acceleration on Jetson Orin Nano.
"""

import logging
import time
import threading
from typing import Optional, Callable, Generator

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Local LLM inference using llama-cpp-python.

    Supports Llama 3.2, Phi-3, Mistral, and other GGUF models.
    """

    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "models/llama-3.2-3b-instruct-q4_k_m.gguf")
        self.n_gpu_layers = config.get("n_gpu_layers", -1)
        self.context_length = config.get("context_length", 2048)
        self.max_tokens = config.get("max_tokens", 256)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.repeat_penalty = config.get("repeat_penalty", 1.1)
        self.system_prompt = config.get(
            "system_prompt",
            "You are a friendly, helpful AI companion. Keep responses concise."
        )

        self._model = None
        self._lock = threading.Lock()
        self._generating = False

    def load(self):
        """Load the LLM model."""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading LLM model: {self.model_path}")
            logger.info(f"  GPU layers: {self.n_gpu_layers}, context: {self.context_length}")

            start = time.time()
            self._model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.context_length,
                n_batch=512,
                verbose=False,
            )
            elapsed = time.time() - start
            logger.info(f"LLM model loaded in {elapsed:.1f}s.")

        except ImportError:
            logger.error(
                "llama-cpp-python not installed. "
                "Build with CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")

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
