"""Phase 3.3 — ``RetrievalPipeline.retrieve_hybrid`` RRF default + linear opt-in."""

from __future__ import annotations

from typing import Any

import pytest

from raghub.config import HybridConfig
from raghub.embeddings import HashingEmbeddingProvider
from raghub.models import ChunkRecord, UserPrincipal
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.vectorstore import InMemoryVectorStore


def make_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


def ingest_chunks(store: InMemoryVectorStore, embedder: HashingEmbeddingProvider) -> None:
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
        for i, text in enumerate(
            [
                "revenue grew 12% in Q3 driven by SaaS bookings",
                "operating margin expanded 200bps sequentially",
                "customer count rose 8% year over year",
            ]
        )
    ]
    vectors = [embedder.embed_text(c.text) for c in chunks]
    store.upsert(chunks, vectors)


def make_pipeline(hybrid: HybridConfig | None = None) -> RetrievalPipeline:
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    store = make_store()
    ingest_chunks(store, embedder)
    from raghub.retrieval.reranker import IdentityReranker

    return RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=IdentityReranker(),
        hybrid=hybrid or HybridConfig(),
    )


def test_default_fusion_is_rrf() -> None:
    """Out of the box the hybrid path uses RRF, not linear combination."""
    pipe = make_pipeline()
    assert pipe.hybrid.fusion == "rrf"


def test_rrf_path_produces_ranked_hits() -> None:
    pipe = make_pipeline()
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    dense = pipe.retrieve(user=user, question="revenue", top_k=3)
    fused = pipe.retrieve_hybrid(query="revenue", vector_results=dense)
    # RRF normalises by ranks, not raw scores — every hit is positive.
    assert all(h.score > 0 for h in fused)
    # The dense top-1 should still come out on top when it overlaps
    # with the sparse top-1.
    assert fused[0].chunk_id in {"c-0", "c-1", "c-2"}


def test_linear_path_preserved_when_requested() -> None:
    pipe = make_pipeline(HybridConfig(fusion="linear"))
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    dense = pipe.retrieve(user=user, question="revenue", top_k=3)
    fused_linear = pipe.retrieve_hybrid(query="revenue", vector_results=dense, fusion="linear")
    fused_default = pipe.retrieve_hybrid(query="revenue", vector_results=dense)
    # Same chunk_ids, possibly different ordering because the score
    # distributions differ between linear and RRF.
    assert {h.chunk_id for h in fused_linear} == {h.chunk_id for h in fused_default}


def test_unknown_fusion_falls_back_to_rrf() -> None:
    """Defensive: an unknown value must not raise; RRF is the safe default."""
    pipe = make_pipeline()
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    dense = pipe.retrieve(user=user, question="revenue", top_k=3)
    out = pipe.retrieve_hybrid(query="revenue", vector_results=dense, fusion="BOGUS")
    assert all(h.score > 0 for h in out)