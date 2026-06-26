"""sentence-transformers fallback embedding provider with 384-dim."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingProvider:
    """sentence-transformers fallback embedding provider with 384-dim."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
