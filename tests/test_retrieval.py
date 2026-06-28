from __future__ import annotations

from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.models import ChunkRecord, Classification, UserPrincipal
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.retrieval.reranker import IdentityReranker
from raghub.retrieval.search import FacetedSearchEngine, SearchFilters, build_filter_string
from raghub.vectorstore.memory import InMemoryVectorStore


def _make_chunk(text: str, company: str = "acme", chunk_id: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id or f"chunk_{hash(text)}",
        text=text,
        document_id="doc1",
        version=1,
        page=1,
        section="test",
        company=company,
        department="eng",
        classification=Classification.INTERNAL,
        owner="user@acme.com",
        metadata={},
    )


class TestBuildFilterString:
    def test_none_returns_empty(self) -> None:
        assert build_filter_string(None) == ""

    def test_empty_filters_returns_empty(self) -> None:
        f = SearchFilters()
        assert build_filter_string(f) == ""

    def test_companies(self) -> None:
        f = SearchFilters(companies=["acme", "beta"])
        result = build_filter_string(f)
        assert "company IN" in result
        assert "'acme'" in result
        assert "'beta'" in result

    def test_owners(self) -> None:
        f = SearchFilters(owners=["alice"])
        result = build_filter_string(f)
        assert "owner IN ('alice')" in result

    def test_file_types(self) -> None:
        f = SearchFilters(file_types=["pdf", "csv"])
        result = build_filter_string(f)
        assert "file_type IN" in result
        assert "'pdf'" in result


class TestFacetedSearchEngine:
    def test_search_returns_chunks(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection()
        provider = HashingEmbeddingProvider(dimension=4)
        engine = FacetedSearchEngine(store, provider)
        chunk = _make_chunk("hello world")
        store.insert([chunk], [provider.embed_text("hello world")])
        results = engine.search("hello")
        assert len(results) == 1
        assert results[0].chunk_id == chunk.chunk_id

    def test_search_with_filters(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection()
        provider = HashingEmbeddingProvider(dimension=4)
        engine = FacetedSearchEngine(store, provider)
        c1 = _make_chunk("alpha", company="acme", chunk_id="c1")
        c2 = _make_chunk("beta", company="beta", chunk_id="c2")
        store.insert([c1, c2], [provider.embed_text("alpha"), provider.embed_text("beta")])
        filters = SearchFilters(companies=["acme"])
        results = engine.search("alpha", filters=filters)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_matches_filters(self) -> None:
        provider = HashingEmbeddingProvider(dimension=4)
        engine = FacetedSearchEngine(InMemoryVectorStore(), provider)
        chunk = _make_chunk("test", company="acme")
        filters = SearchFilters(companies=["acme"], departments=["eng"])
        assert engine.matches_filters(chunk, filters)
        filters2 = SearchFilters(companies=["beta"])
        assert not engine.matches_filters(chunk, filters2)

    def test_count_by_field(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection()
        provider = HashingEmbeddingProvider(dimension=4)
        engine = FacetedSearchEngine(store, provider)
        store.insert(
            [_make_chunk("a", company="acme"), _make_chunk("b", company="beta")],
            [provider.embed_text("a"), provider.embed_text("b")],
        )
        counts = engine.count_by_field("company")
        assert counts.get("acme") == 1
        assert counts.get("beta") == 1


class TestRetrievalPipeline:
    def test_retrieve_returns_deduplicated_hits(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection()
        provider = HashingEmbeddingProvider(dimension=4)
        pipeline = RetrievalPipeline(
            embedding_provider=provider,
            vector_store=store,
            reranker=IdentityReranker(),
        )
        chunk = _make_chunk("hello world")
        store.insert([chunk], [provider.embed_text("hello world")])
        user = UserPrincipal(email="test@acme.com", allowed_companies=["acme"], is_admin=False)
        hits = pipeline.retrieve(user=user, question="hello", top_k=5)
        assert len(hits) == 1
        assert hits[0].chunk_id == chunk.chunk_id

    def test_retrieve_keyword(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection()
        provider = HashingEmbeddingProvider(dimension=4)
        pipeline = RetrievalPipeline(
            embedding_provider=provider,
            vector_store=store,
            reranker=IdentityReranker(),
        )
        chunk = _make_chunk("hello world foo bar")
        store.insert([chunk], [provider.embed_text("hello world foo bar")])
        hits = pipeline.retrieve_keyword("hello", top_k=5)
        assert len(hits) == 1

    def test_retrieve_keyword_empty_records(self) -> None:
        store = InMemoryVectorStore()
        pipeline = RetrievalPipeline(
            embedding_provider=HashingEmbeddingProvider(dimension=4),
            vector_store=store,
            reranker=IdentityReranker(),
        )
        assert pipeline.retrieve_keyword("hello") == []

    def test_hybrid_search(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection()
        provider = HashingEmbeddingProvider(dimension=4)
        pipeline = RetrievalPipeline(
            embedding_provider=provider,
            vector_store=store,
            reranker=IdentityReranker(),
        )
        chunk = _make_chunk("hello world")
        store.insert([chunk], [provider.embed_text("hello world")])
        user = UserPrincipal(email="test@acme.com", allowed_companies=["acme"], is_admin=False)
        hits = pipeline.hybrid_search(user=user, question="hello", top_k=5)
        assert len(hits) >= 1

    def test_retrieve_hybrid_fusion(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection()
        provider = HashingEmbeddingProvider(dimension=4)
        pipeline = RetrievalPipeline(
            embedding_provider=provider,
            vector_store=store,
            reranker=IdentityReranker(),
        )
        c1 = _make_chunk("alpha", chunk_id="c1")
        c2 = _make_chunk("beta", chunk_id="c2")
        store.insert([c1, c2], [provider.embed_text("alpha"), provider.embed_text("beta")])
        user = UserPrincipal(email="test@acme.com", allowed_companies=["acme"], is_admin=False)
        vector_hits = pipeline.retrieve(user=user, question="alpha", top_k=5)
        fused = pipeline.retrieve_hybrid("alpha", vector_hits)
        assert len(fused) > 0
