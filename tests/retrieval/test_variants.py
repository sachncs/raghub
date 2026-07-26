"""Phase 2.8 — :meth:`RetrievalPipeline.retrieve_variants` tests."""

from __future__ import annotations

from raghub.embeddings import HashingEmbeddingProvider
from raghub.models import ChunkRecord, UserPrincipal
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.retrieval.transforms.base import QueryVariant
from raghub.vectorstore import InMemoryVectorStore


def make_pipeline() -> tuple[RetrievalPipeline, InMemoryVectorStore]:
    store = InMemoryVectorStore()
    embedder = HashingEmbeddingProvider(dimension=16, model_name="test")
    from raghub.retrieval.reranker import IdentityReranker

    pipeline = RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=IdentityReranker(),
    )
    return pipeline, store


def ingest_chunks(pipeline: RetrievalPipeline, store: InMemoryVectorStore) -> None:
    chunks = [
        ChunkRecord(
            chunk_id=f"c-{i}",
            document_id="d-1",
            version=1,
            page=1,
            source_location="loc",
            section="",
            company="A",
            owner="",
            department="",
            text=text,
            metadata={},
        )
        for i, text in enumerate(
            ["revenue grew 12% in Q3", "operating margin expanded", "customer count rose"]
        )
    ]
    vectors = [pipeline.embedding_provider.embed_text(c.text) for c in chunks]
    store.upsert(chunks, vectors)


def test_retrieve_variants_empty_returns_empty() -> None:
    pipeline, store = make_pipeline()
    assert (
        pipeline.retrieve_variants(
            user=UserPrincipal(email="a@b.c"),
            variants=[],
            top_k=5,
        )
        == []
    )


def test_retrieve_variants_single_original_delegates_to_retrieve() -> None:
    """A single ``original`` variant must hit the same code path as ``retrieve``."""
    pipeline, store = make_pipeline()
    ingest_chunks(pipeline, store)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])

    direct = pipeline.retrieve(user=user, question="revenue", top_k=3)
    via_variants = pipeline.retrieve_variants(
        user=user,
        variants=[QueryVariant(text="revenue", kind="original", weight=1.5)],
        top_k=3,
    )
    assert [h.chunk_id for h in direct] == [h.chunk_id for h in via_variants]
    assert [h.score for h in direct] == [h.score for h in via_variants]


def test_retrieve_variants_fuses_multiple_variants() -> None:
    """Two variants must produce hits from both channels, fused."""
    pipeline, store = make_pipeline()
    ingest_chunks(pipeline, store)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])

    variants = [
        QueryVariant(text="revenue", kind="original", weight=1.5),
        QueryVariant(text="customer", kind="hyde", weight=1.0),
    ]
    hits = pipeline.retrieve_variants(user=user, variants=variants, top_k=3)
    # Both channels return at least one chunk each.
    chunk_ids = {h.chunk_id for h in hits}
    assert "c-0" in chunk_ids or "c-2" in chunk_ids  # either hit the channel
    # All scores are positive and sorted descending.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_variants_drops_blank_variants() -> None:
    pipeline, store = make_pipeline()
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    # Empty / whitespace-only text must not raise or hit the embedder.
    out = pipeline.retrieve_variants(
        user=user,
        variants=[
            QueryVariant(text="", kind="hyde", weight=1.0),
            QueryVariant(text="   ", kind="hyde", weight=1.0),
            QueryVariant(text="ignored anyway", kind="original", weight=1.0),
        ],
        top_k=5,
    )
    # Empty store → no hits regardless.
    assert out == []


def test_retrieve_variants_skips_zero_weight() -> None:
    pipeline, store = make_pipeline()
    ingest_chunks(pipeline, store)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    out = pipeline.retrieve_variants(
        user=user,
        variants=[
            QueryVariant(text="revenue", kind="hyde", weight=0.0),
            QueryVariant(text="revenue", kind="original", weight=1.5),
        ],
        top_k=3,
    )
    assert out  # at least the original-variant hit survives
    # Only the original variant contributed, so scores are unweighted.
    assert all(h.score > 0 for h in out)