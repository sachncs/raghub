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

import asyncio
import os
from hashlib import sha256
from typing import Any

import litellm
import numpy as np

from raghub.constants import (
    DEFAULT_EMBEDDING_DIM,
    ENV_ANTHROPIC_API_KEY,
    ENV_AZURE_API_KEY,
    ENV_COHERE_API_KEY,
    ENV_LITELLM_API_KEY,
    ENV_NVIDIA_API_KEY,
    ENV_OPENAI_API_KEY,
    ENV_VOYAGE_API_KEY,
    HASHING_BGE_MODEL,
)
from raghub.errors import ConfigurationError
from raghub.registry import Registry

__all__ = [
    "Embedder",
    "FeatureHashingEmbedder",
    "LiteLLMEmbedder",
    "build_embedder",
]


class Embedder(Registry):
    """Polymorphic base for embedding providers.

    Concrete providers register themselves with ``@Embedder.register``
    and implement :meth:`embed_text`. The :meth:`embed_texts` default
    simply calls it once per string, but providers with batched APIs
    (NVIDIA) should override for throughput.
    """

    model_name: str
    dimension: int

    def embed_text(self, text: str) -> list[float]:
        """Embed one string into a fixed-dimension vector."""
        raise NotImplementedError

    async def aembed_text(self, text: str) -> list[float]:
        """Async wrapper around :meth:`embed_text`.

        Default implementation runs the synchronous call in a worker
        thread so the event loop is not blocked. Concrete providers
        with native async backends can override for direct await.
        """
        return await asyncio.to_thread(self.embed_text, text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings.

        Default implementation loops over :meth:`embed_text`. Override
        when the backing API supports batched calls.
        """
        return [self.embed_text(text) for text in texts]


@Embedder.register("hashing")
class FeatureHashingEmbedder(Embedder):
    """Feature-hashing embedder producing deterministic L2-normalised vectors.

    The provider hashes each whitespace-delimited, lower-cased token into a
    fixed-dimension bucket with a random sign. This is the "hashing trick"
    used to compress very-high-cardinality feature spaces; here it stands in
    for a real text embedding model.
    """

    def __init__(
        self, dimension: int = DEFAULT_EMBEDDING_DIM, model_name: str = HASHING_BGE_MODEL
    ) -> None:
        """Initialise the embedder.

        Args:
            dimension: Output vector size. Must be a positive integer;
                larger values reduce bucket collisions at the cost of memory.
            model_name: Stable label exposed via :pyattr:`model_name`.

        """
        self.dimension = dimension
        self.model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        """Hash ``text`` into a deterministic L2-normalised vector."""
        if not text:
            return [0.0] * self.dimension
        return self.embed_texts([text])[0]

    async def aembed_text(self, text: str) -> list[float]:
        """Async variant: hashing is in-process; no I/O to offload."""
        return self.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Hash every input into a deterministic L2-normalised vector."""
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


@Embedder.register("litellm")
class LiteLLMEmbedder(Embedder):
    """Embedding provider backed by LiteLLM."""

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
        """Embed a single string."""
        return self.embed_texts([text])[0]

    async def aembed_text(self, text: str) -> list[float]:
        """Async wrapper: run the LiteLLM call in a worker thread.

        The LiteLLM SDK is sync; calling it directly inside an async
        coroutine would block the event loop, so we offload to a
        thread.
        """
        return await asyncio.to_thread(self.embed_text, text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one LiteLLM call."""
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
            list(record["embedding"]) if isinstance(record, dict) else list(record.embedding)
            for record in data
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
                ENV_OPENAI_API_KEY,
                ENV_ANTHROPIC_API_KEY,
                ENV_NVIDIA_API_KEY,
                ENV_LITELLM_API_KEY,
                ENV_COHERE_API_KEY,
                ENV_VOYAGE_API_KEY,
                ENV_AZURE_API_KEY,
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
