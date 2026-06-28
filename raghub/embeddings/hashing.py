"""Deterministic feature-hashing embedding provider.

This module ships a zero-dependency, fully offline embedder suitable for
unit tests, CI smoke runs, and development environments where reaching an
external embedding API is undesirable or impossible. It is **not** a
semantic-quality embedder; its purpose is to provide stable vector output so
that downstream code paths (chunking, vector stores, retrieval) can be
exercised end-to-end without network access.

Algorithm: signed random projection via feature hashing (Weinberger et al.,
"Feature Hashing for Large Scale Multitask Learning", ICML 2009). For each
input token we compute ``sha256(token)`` and use the first four bytes as the
bucket index and the low bit of the fifth byte as the sign. Repeated tokens
accumulate (with the same sign) in their bucket, and the resulting vector is
L2-normalised so that cosine similarity degenerates to a dot product.

Trade-offs:

* **Pros:** no model download, no network, fully deterministic across
  processes and platforms, trivially fast.
* **Cons:** no notion of synonymy, no context, no positional information.
  Two unrelated tokens hash to unrelated buckets, so semantically similar
  texts do not necessarily yield similar vectors. For real workloads use
  :class:`raghub.embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider`
  or :class:`raghub.embeddings.nvidia.NvidiaEmbeddingProvider`.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from raghub.embeddings.base import BaseEmbeddingProvider


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Feature-hashing embedder producing deterministic L2-normalised vectors.

    The provider hashes each whitespace-delimited, lower-cased token into a
    fixed-dimension bucket with a random sign. This is the "hashing trick"
    used to compress very-high-cardinality feature spaces; here it stands in
    for a real text embedding model.

    Attributes:
        dimension: Output vector dimensionality. Default 384 matches the
            NV-Embed-QA model used in production so downstream cosine
            comparisons are dimensionally compatible.
        model_name: Stable identifier reported as the provider name; useful
            for telemetry and cache keys.
    """

    def __init__(self, dimension: int = 384, model_name: str = "hashing-bge") -> None:
        """Initialise the embedder.

        Args:
            dimension: Output vector size. Must be a positive integer;
                larger values reduce bucket collisions at the cost of memory.
            model_name: Stable label exposed via :pyattr:`model_name`.
        """
        self.dimension = dimension
        self.model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        """Hash ``text`` into a deterministic L2-normalised vector.

        Implementation details:

        * Tokens are lower-cased and split on ASCII whitespace. No
          tokenisation, stemming, or stop-word filtering is applied.
        * Each token contributes a signed ``+/-1`` to its hashed bucket, so
          repeated tokens accumulate linearly in that bucket.
        * The result is L2-normalised so that cosine similarity between two
          vectors reduces to a dot product, simplifying downstream maths.

        Args:
            text: The input text. Empty input returns an all-zero vector.

        Returns:
            A list of ``dimension`` floats. Empty inputs return a zero
            vector (so the caller can distinguish "no signal" from "no
            overlap" by inspecting the norm).
        """
        # Allocate the accumulator once and mutate in place for speed.
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            # No tokens => no signal. Returning the zero vector (rather
            # than a random one) keeps the result deterministic and lets
            # downstream code short-circuit empty-text edge cases.
            return vector.tolist()
        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            # First 4 bytes => 32-bit unsigned int, interpreted little-endian.
            # Modular reduction maps any token to a bucket in [0, dimension).
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            # The fifth byte's LSB supplies a pseudo-random sign. This is
            # the standard "signed hashing" trick from Weinberger et al.:
            # it preserves inner-product estimates in expectation while
            # bounding the variance of any single bucket.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            # Accumulate so repeated tokens reinforce their bucket.
            vector[idx] += sign
        # L2-normalise so cosine similarity == dot product downstream.
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()
