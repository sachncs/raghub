"""Embedding provider base class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider."""

    model_name: str

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed one text."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts."""

        return [self.embed_text(text) for text in texts]

