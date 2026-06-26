"""Embedding adapter used by retrieval and ingestion.

The implementation is intentionally isolated so another provider can be
replaced later without changing business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256

import numpy as np


class Embedder(ABC):
    """Embeds text into vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class HashingEmbedder(Embedder):
    """Deterministic local embedder for offline development."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in text.lower().split():
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

