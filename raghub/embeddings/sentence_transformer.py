"""Sentence-Transformers embedding provider.

Local CPU/GPU embedding via the ``sentence-transformers`` library.
Default model is ``all-MiniLM-L6-v2`` (384-dim). No network calls are
made beyond the one-time model download performed by ``SentenceTransformer``.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from raghub.embeddings.base import BaseEmbeddingProvider


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


__all__ = ["SentenceTransformerEmbeddingProvider"]
