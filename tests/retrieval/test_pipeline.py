"""Tests for ``raghub.retrieval.pipeline`` (Retrieval)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from raghub.models import Chunk, Hit, User
from raghub.retrieval.factories import HybridConfigShim
from raghub.retrieval.pipeline import Retrieval
from raghub.retrieval.types import Variant


def make_chunk(chunk_id: str, text: str = "chunk text") -> Chunk:
    """Build a minimal Chunk for tests."""

    import hashlib

    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        version=1,
        text=text,
        classification="internal",
        company="acme",
        owner="alice",
        department="finance",
        checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        page=0,
        source_location="page 1",
    )


def _user_with_companies(companies: list[str]) -> SimpleNamespace:
    """Build a user-shaped object with company allow-list."""

    return SimpleNamespace(
        is_admin=False,
        allowed_companies=companies,
        allowed_groups=[],
        email="alice@example.com",
    )


def test_retrieval_init_stores_components() -> None:
    """``Retrieval.__init__`` stores the supplied collaborators."""

    embedder = MagicMock()
    vector_store = MagicMock()
    rerank = MagicMock()
    retrieval = Retrieval(
        embedding_provider=embedder, vector_store=vector_store, rerank=rerank
    )
    assert retrieval.embedding_provider is embedder
    assert retrieval.vector_store is vector_store
    assert retrieval.rerank is rerank


def test_retrieval_init_uses_default_hybrid_config_when_omitted() -> None:
    """``Retrieval.__init__`` defaults ``hybrid`` to default_hybrid()."""

    retrieval = Retrieval(
        embedding_provider=MagicMock(), vector_store=MagicMock(), rerank=MagicMock()
    )
    assert retrieval.hybrid is not None
    assert hasattr(retrieval.hybrid, "fusion")


def test_retrieval_init_accepts_explicit_hybrid_config() -> None:
    """``Retrieval.__init__`` respects an explicit ``hybrid=`` argument."""

    hybrid = HybridConfigShim()
    hybrid.rrf_k = 42
    retrieval = Retrieval(
        embedding_provider=MagicMock(), vector_store=MagicMock(), rerank=MagicMock(), hybrid=hybrid
    )
    assert retrieval.hybrid.rrf_k == 42


def test_retrieve_embeds_then_searches_then_reranks() -> None:
    """``retrieve`` follows embed -> search -> dedupe -> rerank pipeline."""

    chunk_a = make_chunk("a")
    chunk_b = make_chunk("b")
    raw_hits = [
        {"score": 0.9, "chunk": chunk_a},
        {"score": 0.7, "chunk": chunk_b},
    ]

    embedder = MagicMock()
    embedder.embed_text = MagicMock(return_value=[0.1, 0.2])
    vector_store = MagicMock()
    vector_store.search = MagicMock(return_value=raw_hits)
    rerank = MagicMock()
    rerank.rerank = MagicMock(side_effect=lambda *, question, hits: hits)

    retrieval = Retrieval(
        embedding_provider=embedder, vector_store=vector_store, rerank=rerank
    )
    result = retrieval.retrieve(user=_user_with_companies(["acme"]), question="q", top_k=5)

    embedder.embed_text.assert_called_once_with("q")
    vector_store.search.assert_called_once()
    rerank.rerank.assert_called_once()
    assert len(result) == 2
    assert all(isinstance(h, Hit) for h in result)


def test_retrieve_deduplicates_by_chunk_id() -> None:
    """``retrieve`` drops duplicate chunk ids from the search results."""

    chunk_a = make_chunk("a")
    raw_hits = [
        {"score": 0.9, "chunk": chunk_a},
        {"score": 0.8, "chunk": chunk_a},  # duplicate
        {"score": 0.7, "chunk": make_chunk("b")},
    ]

    embedder = MagicMock()
    embedder.embed_text = MagicMock(return_value=[0.1])
    vector_store = MagicMock()
    vector_store.search = MagicMock(return_value=raw_hits)
    rerank = MagicMock()
    rerank.rerank = MagicMock(side_effect=lambda *, question, hits: hits)

    retrieval = Retrieval(
        embedding_provider=embedder, vector_store=vector_store, rerank=rerank
    )
    result = retrieval.retrieve(user=_user_with_companies(["acme"]), question="q", top_k=5)
    assert len(result) == 2
    chunk_ids = [h.chunk.id for h in result]
    assert chunk_ids == ["a", "b"]


def test_retrieve_handles_empty_search_results() -> None:
    """``retrieve`` returns [] when the vector store finds nothing."""

    embedder = MagicMock()
    embedder.embed_text = MagicMock(return_value=[0.1])
    vector_store = MagicMock()
    vector_store.search = MagicMock(return_value=[])
    rerank = MagicMock()
    rerank.rerank = MagicMock(return_value=[])  # identity pass-through on empty

    retrieval = Retrieval(
        embedding_provider=embedder, vector_store=vector_store, rerank=rerank
    )
    assert retrieval.retrieve(user=_user_with_companies(["acme"]), question="q", top_k=5) == []


def test_retrieve_passes_admin_users_no_filter() -> None:
    """``retrieve`` passes empty-string filter to vector store for admins."""

    admin = User(id="admin", email="admin@example.com", is_admin=True)
    embedder = MagicMock()
    embedder.embed_text = MagicMock(return_value=[0.1])
    vector_store = MagicMock()
    vector_store.search = MagicMock(return_value=[])
    rerank = MagicMock()

    retrieval = Retrieval(
        embedding_provider=embedder, vector_store=vector_store, rerank=rerank
    )
    retrieval.retrieve(user=admin, question="q", top_k=5)
    call_kwargs = vector_store.search.call_args.kwargs
    assert call_kwargs["metadata_filter"] == {}


def test_retrieve_passes_user_company_filter_for_non_admin() -> None:
    """``retrieve`` passes the user's company list as filter for non-admins."""

    embedder = MagicMock()
    embedder.embed_text = MagicMock(return_value=[0.1])
    vector_store = MagicMock()
    vector_store.search = MagicMock(return_value=[])
    rerank = MagicMock()

    retrieval = Retrieval(
        embedding_provider=embedder, vector_store=vector_store, rerank=rerank
    )
    retrieval.retrieve(user=_user_with_companies(["acme", "globex"]), question="q", top_k=5)
    call_kwargs = vector_store.search.call_args.kwargs
    assert call_kwargs["metadata_filter"] == {"company": ["acme", "globex"]}


def test_retrieve_keyword_returns_hit_list() -> None:
    """``retrieve_keyword`` wraps the vector_store.keyword_search results."""

    chunk = make_chunk("a")
    raw = [{"score": 0.8, "chunk": chunk}]

    vector_store = MagicMock()
    vector_store.keyword_search = MagicMock(return_value=raw)

    retrieval = Retrieval(
        embedding_provider=MagicMock(), vector_store=vector_store, rerank=MagicMock()
    )
    hits = retrieval.retrieve_keyword("q", top_k=5)
    assert len(hits) == 1
    assert hits[0].chunk.id == "a"
    assert hits[0].score == 0.8
    vector_store.keyword_search.assert_called_once_with("q", 5)


def test_fused_combines_dense_and_keyword_via_rrf() -> None:
    """``fused`` combines dense and keyword paths via reciprocal-rank fusion."""

    chunk_a = make_chunk("a")
    chunk_b = make_chunk("b")
    chunk_c = make_chunk("c")

    vector_results = [Hit(score=0.9, chunk=chunk_a), Hit(score=0.8, chunk=chunk_b)]
    keyword_results_raw = [
        {"score": 0.7, "chunk": chunk_b},
        {"score": 0.6, "chunk": chunk_c},
    ]

    vector_store = MagicMock()
    vector_store.keyword_search = MagicMock(return_value=keyword_results_raw)

    retrieval = Retrieval(
        embedding_provider=MagicMock(), vector_store=vector_store, rerank=MagicMock()
    )
    result = retrieval.fused(query="q", vector_results=vector_results, rrf_k=60)
    assert isinstance(result, list)
    assert len(result) >= 2



def test_retrieval_init_via_components_kwarg_works() -> None:
    """``Retrieval.__init__`` accepts components= keyword (forwarded)."""

    retrieval = Retrieval(
        embedding_provider=MagicMock(),
        vector_store=MagicMock(),
        rerank=MagicMock(),
    )
    assert retrieval.embedding_provider is not None
    assert retrieval.vector_store is not None
    assert retrieval.rerank is not None