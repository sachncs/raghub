from __future__ import annotations

from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from raghub.embeddings import build_embedding_provider


class TestHashingEmbeddingProvider:
    def test_embed_text_returns_vector(self) -> None:
        provider = HashingEmbeddingProvider(dimension=4)
        vec = provider.embed_text("hello")
        assert len(vec) == 4

    def test_dimension_property(self) -> None:
        provider = HashingEmbeddingProvider(dimension=8)
        assert provider.dimension == 8


class TestSentenceTransformerEmbeddingProvider:
    def test_embed_returns_vectors(self) -> None:
        provider = SentenceTransformerEmbeddingProvider()
        result = provider.embed_texts(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == provider.dimension

    def test_embed_query(self) -> None:
        provider = SentenceTransformerEmbeddingProvider()
        vec = provider.embed_text("test query")
        assert len(vec) == provider.dimension

    def test_dimension_property(self) -> None:
        provider = SentenceTransformerEmbeddingProvider()
        assert provider.dimension > 0


class TestBuildEmbeddingProvider:
    def test_returns_hashing_for_hashing_model(self) -> None:
        provider = build_embedding_provider("hashing-bge", 384)
        assert isinstance(provider, HashingEmbeddingProvider)

    def test_returns_sentence_transformer_for_other(self) -> None:
        provider = build_embedding_provider("all-MiniLM-L6-v2", 384)
        assert isinstance(provider, SentenceTransformerEmbeddingProvider)
