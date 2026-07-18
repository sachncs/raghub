"""Embedding provider base class."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
