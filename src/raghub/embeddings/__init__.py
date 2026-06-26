"""Embedding providers."""

from .base import BaseEmbeddingProvider
from .hashing import HashingEmbeddingProvider
from .nvidia import NvidiaEmbeddingProvider
from .sentence_transformer import SentenceTransformerEmbeddingProvider


def build_embedding_provider(
    model_name: str,
    dimension: int,
    api_key: str | None = None,
) -> NvidiaEmbeddingProvider | SentenceTransformerEmbeddingProvider | HashingEmbeddingProvider:
    if "nvidia" in model_name.lower() or api_key is not None:
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
