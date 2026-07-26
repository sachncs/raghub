"""Phase 3.1 + 3.4 + 3.5 — BM25 / ColBERT / hybrid v2 tests."""

from __future__ import annotations

from typing import Any

import pytest

from raghub.config.settings import HybridConfig
from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.exceptions import GraphUnavailableError
from raghub.models import ChunkRecord, UserPrincipal
from raghub.retrieval.colbert import ColbertLateInteraction
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.vectorstore.memory import InMemoryVectorStore


# --- 3.1: BM25 -------------------------------------------------------------


def store_with_texts(texts: list[str]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    chunks = [
        ChunkRecord(
            chunk_id=f"c-{i}",
            document_id="d",
            version=1,
            page=1,
            source_location="s",
            section="",
            company="A",
            owner="",
            department="",
            text=text,
            metadata={},
        )
        for i, text in enumerate(texts)
    ]
    store.upsert(chunks, [[0.0] * 4 for _ in chunks])
    return store


def test_keyword_search_uses_bm25_when_available() -> None:
    pytest.importorskip("rank_bm25")
    # 4-doc corpus so BM25's IDF is well-behaved (2-doc corpora
    # give a negative log that rank-bm25 clamps to 0).
    store = store_with_texts(
        [
            "revenue grew 12% in Q3",
            "operating margin expanded",
            "apple pie is delicious",
            "banana split is cold",
        ]
    )
    hits = store.keyword_search("revenue", top_k=4)
    assert len(hits) == 1
    # BM25 score for the matching chunk should be a positive float.
    assert hits[0]["score"] > 0
    assert hits[0]["chunk_id"] == "c-0"


def test_keyword_search_tf_fallback_when_bm25_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulated ImportError on rank_bm25 falls back to the TF path."""
    import builtins

    from raghub.vectorstore import memory as memory_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rank_bm25" or name.startswith("rank_bm25"):
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Force the module-level cache to forget the (possibly present) import.
    monkeypatch.setattr(memory_mod, "BM25Okapi", None, raising=False)

    store = store_with_texts(["revenue grew 12% in Q3", "operating margin expanded"])
    hits = store.keyword_search("revenue", top_k=2)
    # TF fallback: 1 match per query term per chunk, divided by chunk
    # length. The matching chunk wins, the other gets 0.
    assert [h["chunk_id"] for h in hits] == ["c-0"]


# --- 3.4: ColBERT adapter ---------------------------------------------------


def test_colbert_disabled_by_default() -> None:
    cfg = HybridConfig()
    adapter = ColbertLateInteraction(cfg)
    assert adapter.is_available() is False


def test_colbert_enabled_but_missing_dependency_raises() -> None:
    """``is_available()`` returns ``False`` even when the flag is on,
    when the optional dep is not installed. ``score()`` raises
    :class:`GraphUnavailableError` to surface the misconfig.
    """
    cfg = HybridConfig(colbert_enabled=True)
    adapter = ColbertLateInteraction(cfg)
    assert adapter.is_available() is False
    with pytest.raises(GraphUnavailableError):
        adapter.score("q", ["doc"])


# --- 3.5: hybrid v2 --------------------------------------------------------


def make_pipeline(hybrid: HybridConfig | None = None) -> tuple[RetrievalPipeline, Any]:
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    store = store_with_texts(
        [
            "revenue grew 12% in Q3",
            "operating margin expanded",
            "customer count rose",
        ]
    )
    chunks = [rec.chunk for rec in store.records.values()]
    vectors = [embedder.embed_text(c.text) for c in chunks]
    store.upsert(chunks, vectors)  # attach the embeddings
    from raghub.retrieval.reranker import IdentityReranker

    return RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=IdentityReranker(),
        hybrid=hybrid or HybridConfig(),
    ), store


def test_retrieve_hybrid_v2_runs_without_colbert() -> None:
    pipe, store = make_pipeline()
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    out = pipe.retrieve_hybrid_v2(
        user=user, question="revenue", top_k=3, colbert=None
    )
    # Dense + sparse channels, RRF-fused.
    assert out
    assert all(h.score > 0 for h in out)


def test_retrieve_hybrid_v2_skips_unavailable_colbert() -> None:
    """An unavailable ColBERT adapter is silently dropped."""
    pipe, store = make_pipeline()
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    adapter = ColbertLateInteraction(HybridConfig(colbert_enabled=True))
    assert adapter.is_available() is False
    out = pipe.retrieve_hybrid_v2(
        user=user, question="revenue", top_k=3, colbert=adapter
    )
    # No exception; RRF still produces a non-empty fused list.
    assert out


def test_retrieve_hybrid_v2_empty_everything() -> None:
    pipe, store = make_pipeline()
    user = UserPrincipal(email="a@b.c", allowed_companies=["ZZ"])  # RBAC blocks all
    out = pipe.retrieve_hybrid_v2(user=user, question="anything", top_k=3)
    assert out == []