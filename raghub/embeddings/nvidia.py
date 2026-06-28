"""NVIDIA embedding provider via :mod:`langchain_nvidia_ai_endpoints`.

Production embedding model: ``nvidia/nv-embed-qa`` (default 384-dim).
Each embedding call is wrapped in the standard retry helper so
transient upstream errors are absorbed.
"""

from __future__ import annotations

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.utils.retry import retry


class NvidiaEmbeddingProvider(BaseEmbeddingProvider):
    """NV-Embed-QA embedding provider via NVIDIA's hosted endpoints."""

    def __init__(
        self,
        model: str = "nvidia/nv-embed-qa",
        dimension: int = 384,
        api_key: str | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            model: NVIDIA model id. Default ``"nvidia/nv-embed-qa"``.
            dimension: Output vector dimensionality. Default 384
                matches the model default and is compatible with the
                hashing embedder's default for mixed-batch scenarios.
            api_key: Optional NVIDIA API key. Falls back to the
                ``NVIDIA_API_KEY`` env var when ``None``.
        """
        self.model_name = model
        self._dimension = dimension
        self.client = NVIDIAEmbeddings(model=model, dims=dimension, api_key=api_key or None)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text via NVIDIA's hosted model.

        Args:
            text: The input text.

        Returns:
            A ``dimension``-sized float vector.
        """
        return retry(lambda: self.client.embed_query(text))  # type: ignore[return-value]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in one batched call.

        Args:
            texts: The input texts.

        Returns:
            A list of ``dimension``-sized float vectors, one per input.
        """
        return retry(lambda: self.client.embed_documents(texts))  # type: ignore[return-value]

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension