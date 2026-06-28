"""Embedding providers.

This package ships:

* :class:`BaseEmbeddingProvider` — abstract base class.
* :class:`HashingEmbeddingProvider` — zero-dependency deterministic
  embedder backed by feature hashing.
* :class:`NvidiaEmbeddingProvider` — production NVIDIA embedder.
* :class:`SentenceTransformerEmbeddingProvider` — local SentenceTransformers
  embedder.

:func:`build_embedding_provider` chooses an implementation from the
model name: substring ``"nvidia"`` → NVIDIA, ``"hashing"`` →
hash-based, otherwise SentenceTransformers.
"""

from .base import BaseEmbeddingProvider
from .hashing import HashingEmbeddingProvider
from .nvidia import NvidiaEmbeddingProvider
from .sentence_transformer import SentenceTransformerEmbeddingProvider


def build_embedding_provider(
    model_name: str,
    dimension: int,
    api_key: str | None = None,
) -> NvidiaEmbeddingProvider | SentenceTransformerEmbeddingProvider | HashingEmbeddingProvider:
    """Construct the appropriate embedding provider for ``model_name``.

    Args:
        model_name: Model identifier; matched case-insensitively
            against the substrings ``"nvidia"`` and ``"hashing"``.
        dimension: Output vector dimensionality; passed through to
            the provider.
        api_key: Optional NVIDIA API key (ignored by the hashing and
            SentenceTransformer providers).

    Returns:
        A ready-to-use embedding provider instance.
    """
    if "nvidia" in model_name.lower():
        return NvidiaEmbeddingProvider(model=model_name, dimension=dimension, api_key=api_key)
    if "hashing" in model_name.lower():
        return HashingEmbeddingProvider(dimension=dimension, model_name=model_name)
    return SentenceTransformerEmbeddingProvider(model_name=model_name)


__all__ = [
    "BaseEmbeddingProvider",
    "HashingEmbeddingProvider",
    "NvidiaEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "build_embedding_provider",
]