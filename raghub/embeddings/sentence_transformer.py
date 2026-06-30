"""Sentence-Transformers embedding provider.

Local CPU/GPU embedding via the ``sentence-transformers`` library.
Default model is ``all-MiniLM-L6-v2`` (384-dim). No network calls are
made beyond the one-time model download performed by ``SentenceTransformer``.
"""

from __future__ import annotations

from typing import Any

from raghub.embeddings.base import BaseEmbeddingProvider

SentenceTransformer: Any

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
    OptionalImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    SentenceTransformer = None
    ST_AVAILABLE = False
    OptionalImportError = exc


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Sentence-Transformers-backed embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the SentenceTransformer model.

        Args:
            model_name: HuggingFace model id. The default
                ``all-MiniLM-L6-v2`` produces 384-dim embeddings.
        """
        if not ST_AVAILABLE:
            from raghub.exceptions import ConfigurationError

            raise ConfigurationError(
                "sentence-transformers is not installed; run "
                "`pip install sentence-transformers`."
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text via SentenceTransformer's batched API.

        Args:
            text: The input text.

        Returns:
            A 384-dim (or model-specific dim) float vector.
        """
        return self.model.encode([text]).tolist()[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in one batched call.

        Args:
            texts: The input texts.

        Returns:
            A list of float vectors, one per input.
        """
        return self.model.encode(texts).tolist()

    @property
    def dimension(self) -> int:
        """Return the model's native embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
