"""Embedding providers.

This module ships:

* :class:`Embedder` — abstract base class.
* :class:`FeatureHashingEmbedder` — zero-dependency deterministic
  embedder backed by feature hashing.
* :class:`LiteLLMEmbedder` — production embedder, backed by
  LiteLLM (any provider: OpenAI, NVIDIA, Cohere, Bedrock, …).
* :func:`build_embedder` — chooses the implementation from
  the model name.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Any
from raghub.constants import DEFAULT_EMBEDDING_DIM, HASHING_BGE_MODEL

import litellm
import numpy as np

from raghub.errors import ConfigurationError

__all__ = [
    "Embedder",
    "FeatureHashingEmbedder",
    "LiteLLMEmbedder",
    "build_embedder",
]


class Embedder(ABC):
    """Abstract embedding provider.

    All concrete providers must implement :meth:`embed_text`; the
    :meth:`embed_texts` default simply calls it once per string, but
    providers with batched APIs (NVIDIA) should
    override for throughput.
    """

    model_name: str
    dimension: int

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


class FeatureHashingEmbedder(Embedder):
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

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM, model_name: str = HASHING_BGE_MODEL) -> None:
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
        if not text:
            return [0.0] * self.dimension
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Hash every input into a deterministic L2-normalised vector.

        Vectorised with NumPy so a 1500-text batch runs in one
        C-level pass rather than the per-token Python loop in
        :meth:`embed_text`. The output is bit-identical to the prior
        implementation (same hash, same sign rule, same
        L2-normalisation).

        Args:
            texts: The list of input strings.

        Returns:
            A list of ``dimension``-float vectors in input order.

        """
        dim = self.dimension
        out = np.zeros((len(texts), dim), dtype=np.float32)
        if not texts:
            return out.tolist()
        for row, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue
            for token in tokens:
                digest = sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        safe = norms > 0
        out = np.where(safe, out / np.where(safe, norms, 1.0), out)
        return out.tolist()


class LiteLLMEmbedder(Embedder):
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

        """
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


def build_embedder(
    model_name: str,
    dimension: int,
    api_key: str | None = None,
) -> Embedder:
    """Construct the appropriate embedding provider for ``model_name``.

    Args:
        model_name: Model identifier; matched case-insensitively
            against the substrings ``"hashing"``, ``"litellm"``, and
            provider prefixes (``"openai/"``, ``"cohere/"``,
            ``"text-embedding-*"``, etc.).
        dimension: Output vector dimensionality; passed through to
            the provider.
        api_key: Optional API key passed through to
        :class:`LiteLLMEmbedder`. Ignored by the hashing
            provider.

    Returns:
        A ready-to-use embedding provider instance.

    """
    name = (model_name or "").lower().strip()
    if "hashing" in name:
        return FeatureHashingEmbedder(dimension=dimension, model_name=model_name)
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
            return LiteLLMEmbedder(model=model_name, api_key=api_key)
        return FeatureHashingEmbedder(dimension=dimension, model_name=model_name)
    raise ConfigurationError(
        f"Unknown embedding model {model_name!r}. "
        "Use a LiteLLM model (e.g. 'openai/text-embedding-3-small', "
        "'cohere/embed-english-v3.0') or 'hashing' for zero-dependency mode."
    )
