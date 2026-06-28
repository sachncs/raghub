"""Fast local embedding provider used for offline development and tests."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from raghub.embeddings.base import BaseEmbeddingProvider


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic bag-of-words style embedder."""

    def __init__(self, dimension: int = 384, model_name: str = "hashing-bge") -> None:
        self.dimension = dimension
        self.model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vector.tolist()
        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

