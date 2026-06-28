"""Embedding provider contract.

Structural type used wherever the codebase needs an embedding
provider. Concrete implementations include the hashing, NVIDIA, and
sentence-transformers providers in :mod:`raghub.embeddings`.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Embeds text into a fixed-dimensional vector.

    Attributes:
        model_name: Stable model identifier, e.g.
            ``"hashing-bge"`` or ``"nvidia/nv-embed-qa"``.
    """

    model_name: str

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: The input text.

        Returns:
            A fixed-dimensional float vector.
        """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings.

        Args:
            texts: The input texts.

        Returns:
            A list of fixed-dimensional float vectors, one per input.
        """
