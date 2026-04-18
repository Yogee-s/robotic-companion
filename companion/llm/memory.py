"""Persistent memory — Mem0 + local ChromaDB.

Per-speaker memory collections: "Yogee" has their own scope, "Mom" has
theirs, etc. Writing is cheap (after each assistant reply we extract a
one-line memory if the turn was non-trivial); reading is a top-K vector
search gated on confidence.

Mem0 wraps Chroma under the hood and handles embeddings + dedup, but we
tolerate Mem0 being unavailable at runtime and gracefully degrade to "no
memory" — conversation still works.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


class MemoryStore:
    def __init__(
        self,
        chroma_dir: str,
        enabled: bool = True,
        top_k: int = 3,
        max_entries_per_speaker: int = 500,
    ) -> None:
        self.chroma_dir = chroma_dir
        self.enabled = enabled
        self.top_k = top_k
        self.max_entries = max_entries_per_speaker
        self._client = None
        if self.enabled:
            self._init_backend()

    def _init_backend(self) -> None:
        try:
            from mem0 import Memory  # type: ignore
        except ImportError:
            log.warning("mem0ai not installed — memory disabled.")
            self.enabled = False
            return

        os.makedirs(self.chroma_dir, exist_ok=True)
        try:
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "companion",
                        "path": self.chroma_dir,
                    },
                }
            }
            self._client = Memory.from_config(config)
            log.info(f"Memory store ready at {self.chroma_dir}")
        except Exception as exc:
            log.warning(f"Memory backend init failed: {exc!r} — disabled.")
            self.enabled = False
            self._client = None

    @property
    def available(self) -> bool:
        return self.enabled and self._client is not None

    def add(self, text: str, speaker: str) -> None:
        if not self.available or not text.strip():
            return
        try:
            assert self._client is not None
            self._client.add(text, user_id=self._scope(speaker))
        except Exception as exc:
            log.debug(f"Memory add failed: {exc!r}")

    def retrieve(self, query: str, speaker: str) -> list[str]:
        if not self.available or not query.strip():
            return []
        try:
            assert self._client is not None
            results = self._client.search(query, user_id=self._scope(speaker), limit=self.top_k)
            # Mem0 returns either a list of dicts or {"results": [...]}
            if isinstance(results, dict):
                results = results.get("results", [])
            return [r.get("memory", r.get("text", "")) for r in results if r]
        except Exception as exc:
            log.debug(f"Memory retrieve failed: {exc!r}")
            return []

    def forget(self, speaker: str) -> None:
        if not self.available:
            return
        try:
            assert self._client is not None
            self._client.delete_all(user_id=self._scope(speaker))
        except Exception as exc:
            log.debug(f"Memory forget failed: {exc!r}")

    def _scope(self, speaker: Optional[str]) -> str:
        return speaker or "unknown"
