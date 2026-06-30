"""Embedding providers.

This package ships:

* :class:`BaseEmbeddingProvider` — abstract base class.
* :class:`HashingEmbeddingProvider` — zero-dependency deterministic
  embedder backed by feature hashing.
* :class:`LiteLLMEmbeddingProvider` — production embedder, backed by
  LiteLLM (any provider: OpenAI, NVIDIA, Cohere, Bedrock, …).
* :class:`SentenceTransformerEmbeddingProvider` — local SentenceTransformers
  embedder.

:func:`build_embedding_provider` chooses the implementation from the
model name: substring ``"hashing"`` → hashing, ``"litellm"`` /
provider-prefixed names → LiteLLM, otherwise SentenceTransformers.

``LiteLLMEmbeddingProvider`` and ``SentenceTransformerEmbeddingProvider``
are imported lazily so the base package does not require the
optional SDKs at import time.
"""

from typing import TYPE_CHECKING, Any

from .base import BaseEmbeddingProvider
from .hashing import HashingEmbeddingProvider

if TYPE_CHECKING:
    from .litellm import LiteLLMEmbeddingProvider as LiteLLMEmbeddingProvider
    from .sentence_transformer import (
        SentenceTransformerEmbeddingProvider as SentenceTransformerEmbeddingProvider,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose providers whose SDKs may not be installed."""
    if name == "LiteLLMEmbeddingProvider":
        from .litellm import LiteLLMEmbeddingProvider as _Provider

        return _Provider
    if name == "SentenceTransformerEmbeddingProvider":
        from .sentence_transformer import (
            SentenceTransformerEmbeddingProvider as _Provider,
        )

        return _Provider
    raise AttributeError(f"module 'raghub.embeddings' has no attribute {name!r}")


def build_embedding_provider(
    model_name: str,
    dimension: int,
    api_key: str | None = None,
) -> Any:
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
    import os

    name = (model_name or "").lower().strip()
    if "hashing" in name:
        return HashingEmbeddingProvider(dimension=dimension, model_name=model_name)
    # ``nvidia/`` and similar remote prefixes need a credential. When
    # none is present we silently swap to the local SentenceTransformer
    # so the framework keeps working offline.
    needs_remote = (
        "litellm" in name
        or any(
            name.startswith(prefix)
            for prefix in (
                "openai/",
                "cohere/",
                "voyage/",
                "azure/",
                "nvidia/",
            )
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
            from .litellm import LiteLLMEmbeddingProvider

            return LiteLLMEmbeddingProvider(model=model_name, api_key=api_key)
        return HashingEmbeddingProvider(dimension=dimension, model_name=model_name)
    from .sentence_transformer import SentenceTransformerEmbeddingProvider

    return SentenceTransformerEmbeddingProvider(model_name=model_name)


__all__ = [
    "BaseEmbeddingProvider",
    "HashingEmbeddingProvider",
    "LiteLLMEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "build_embedding_provider",
]
