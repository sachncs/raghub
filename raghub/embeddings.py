"""Embedding providers.

This module ships:

* :class:`BaseEmbeddingProvider` — abstract base class.
* :class:`HashingEmbeddingProvider` — zero-dependency deterministic
  embedder backed by feature hashing.
* :class:`LiteLLMEmbeddingProvider` — production embedder, backed by
  LiteLLM (any provider: OpenAI, NVIDIA, Cohere, Bedrock, …).
* :class:`SentenceTransformerEmbeddingProvider` — local
  SentenceTransformers embedder.
* :func:`build_embedding_provider` — chooses the implementation from
  the model name.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Any

import litellm
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from raghub.exceptions import ConfigurationError


# Module-level flag retained so existing tests that patch
# ``raghub.embeddings.LITELLM_AVAILABLE = False`` can simulate a
# missing optional dependency even though the package is now required.
LITELLM_AVAILABLE = True


class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider.

    All concrete providers must implement :meth:`embed_text`; the
    :meth:`embed_texts` default simply calls it once per string, but
    providers with batched APIs (NVIDIA, sentence-transformers) should
    override for throughput.
    """

    model_name: str

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed one string into a fixed-dimension vector.

        Args:
            text: The input text.

        Returns:
            A list of floats representing the embedding.
        """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings.

        Default implementation loops over :meth:`embed_text`. Override
        when the backing API supports batched calls.

        Args:
            texts: The list of input strings.

        Returns:
            A list of embeddings, one per input string.
        """
        return [self.embed_text(text) for text in texts]


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

        Args:
            text: The input text. Empty input returns an all-zero vector.

        Returns:
            A list of ``dimension`` floats. Empty inputs return a zero
            vector (so the caller can distinguish "no signal" from "no
            overlap" by inspecting the norm).
        """
        vector: NDArray[np.float32] = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return [float(value) for value in vector]
        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return [float(value) for value in vector]


class LiteLLMEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by LiteLLM."""

    model_name: str

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            model: LiteLLM model name (provider-prefixed when needed).
            api_key: Optional API key override.
            api_base: Optional API base override.

        Raises:
            ConfigurationError: When ``litellm`` is not installed.
        """
        if not LITELLM_AVAILABLE:
            raise ConfigurationError("litellm is not installed; run `pip install litellm`.")
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: The input text.

        Returns:
            A float vector.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one LiteLLM call.

        Args:
            texts: The list of input strings.

        Returns:
            A list of float vectors.
        """
        if not texts:
            return []
        kwargs: dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = litellm.embedding(**kwargs)
        data = response.get("data", []) if isinstance(response, dict) else response.data
        return [
            list(item["embedding"]) if isinstance(item, dict) else list(item.embedding)
            for item in data
        ]


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Sentence-Transformers-backed embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the SentenceTransformer model.

        Args:
            model_name: HuggingFace model id. The default
                ``all-MiniLM-L6-v2`` produces 384-dim embeddings.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_single(self, text: str) -> list[float]:
        """Embed a single text. Returns float vector."""
        return list(self.model.encode([text])[0])

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [list(v) for v in self.model.encode(texts)]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text via SentenceTransformer's batched API.

        Args:
            text: The input text.

        Returns:
            A 384-dim (or model-specific dim) float vector.
        """
        return self.encode_single(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in one batched call.

        Args:
            texts: The input texts.

        Returns:
            A list of float vectors, one per input.
        """
        return self.encode_batch(texts)

    @property
    def dimension(self) -> int:
        """Return the model's native embedding dimension."""
        return int(self.model.get_sentence_embedding_dimension())


def build_embedding_provider(
    model_name: str,
    dimension: int,
    api_key: str | None = None,
) -> BaseEmbeddingProvider:
    """Construct the appropriate embedding provider for ``model_name``.

    Args:
        model_name: Model identifier; matched case-insensitively
            against the substrings ``"hashing"``, ``"litellm"``, and
            provider prefixes (``"openai/"``, ``"cohere/"``,
            ``"text-embedding-*"``, etc.).
        dimension: Output vector dimensionality; passed through to
            the provider.
        api_key: Optional API key passed through to
        :class:`LiteLLMEmbeddingProvider`. Ignored by the hashing
            and SentenceTransformer providers.

    Returns:
        A ready-to-use embedding provider instance.
    """
    name = (model_name or "").lower().strip()
    if "hashing" in name:
        return HashingEmbeddingProvider(dimension=dimension, model_name=model_name)
    needs_remote = "litellm" in name or any(
        name.startswith(prefix)
        for prefix in (
            "openai/",
            "cohere/",
            "voyage/",
            "azure/",
            "nvidia/",
        )
    )
    if needs_remote:
        creds_present = bool(api_key) or any(
            os.getenv(k)
            for k in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "NVIDIA_API_KEY",
                "LITELLM_API_KEY",
                "COHERE_API_KEY",
                "VOYAGE_API_KEY",
                "AZURE_API_KEY",
            )
        )
        if creds_present:
            return LiteLLMEmbeddingProvider(model=model_name, api_key=api_key)
        return HashingEmbeddingProvider(dimension=dimension, model_name=model_name)
    return SentenceTransformerEmbeddingProvider(model_name=model_name)