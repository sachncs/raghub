"""sentence-transformers fallback embedding provider with 384-dim."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from raghub.embeddings.base import BaseEmbeddingProvider


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """sentence-transformers fallback embedding provider with 384-dim."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode([text]).tolist()[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
