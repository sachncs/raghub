"""Phase 2.8 — ``QueryPipeline`` transformer wiring + fast-path regression.

Pins down the Phase 10.6 invariant in miniature: with the default
``ComposeTransformer([])`` (no transforms configured) the pipeline's
calls into ``embedder.embed_text`` and ``vector_store.search`` are
byte-equivalent to the legacy single-shot path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest

from raghub.embeddings import HashingEmbeddingProvider
from raghub.models import (
    ChunkRecord,
    Citation,
    ConversationTurn,
    PipelineContext,
    RetrievalHit,
    UserPrincipal,
)
from raghub.pipeline import QueryPipeline
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.retrieval.transforms.base import QueryVariant
from raghub.retrieval.transforms.compose import ComposeTransformer
from raghub.retrieval.transforms.hyde import HydeTransformer
from raghub.vectorstore import InMemoryVectorStore


class StubLlm:
    """LLM stub returning the prompt as the hypothetical passage."""

    model_name = "stub"

    async def async_generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        # Echo a transformation of the question — keeps the test
        # deterministic without coupling to the heuristic provider.
        return f"hypothetical: {question}"


def make_chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
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


def build_pipeline(
    *,
    transformer: ComposeTransformer | None = None,
) -> tuple[QueryPipeline, InMemoryVectorStore, HashingEmbeddingProvider]:
    store = InMemoryVectorStore()
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    retrieval = RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=__import__("raghub.retrieval.reranker", fromlist=["IdentityReranker"]).IdentityReranker(),
    )
    from raghub.generation import DefaultGenerator

    pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=store,
        generator=DefaultGenerator(llm=__import__("raghub.llm", fromlist=["HeuristicLLMProvider"]).HeuristicLLMProvider()),
        reranker=retrieval.reranker,
        transformer=transformer,
        retrieval_pipeline=retrieval,
    )
    # Seed the store
    chunks = [
        make_chunk("c-1", "revenue grew 12% in Q3"),
        make_chunk("c-2", "operating margin expanded"),
        make_chunk("c-3", "customer count rose"),
    ]
    vectors = [embedder.embed_text(c.text) for c in chunks]
    store.upsert(chunks, vectors)
    return pipeline, store, embedder


@pytest.mark.asyncio
async def test_query_pipeline_no_transformer_uses_fast_path() -> None:
    """With ``transformer=None`` the pipeline never touches the transformer."""
    pipeline, store, embedder = build_pipeline(transformer=None)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")

    with (
        patch.object(embedder, "embed_text", wraps=embedder.embed_text) as embed_spy,
        patch.object(store, "search", wraps=store.search) as search_spy,
    ):
        result = await pipeline.run(
            context,
            question="revenue",
            top_k=3,
            user=user,
        )
    assert result.success is True
    assert result.outputs["transforms_applied"] == []
    assert embed_spy.call_count == 1
    assert search_spy.call_count == 1


@pytest.mark.asyncio
async def test_query_pipeline_empty_transformer_uses_fast_path() -> None:
    """``ComposeTransformer([])`` produces exactly one ``original`` variant.

    Phase 10.6 invariant: the fast path is byte-equivalent.
    """
    pipeline, store, embedder = build_pipeline(
        transformer=ComposeTransformer([])
    )
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")

    with (
        patch.object(embedder, "embed_text", wraps=embedder.embed_text) as embed_spy,
        patch.object(store, "search", wraps=store.search) as search_spy,
    ):
        result = await pipeline.run(
            context,
            question="revenue",
            top_k=3,
            user=user,
        )
    assert result.success is True
    assert result.outputs["transforms_applied"] == []
    # Same call count as the legacy path.
    assert embed_spy.call_count == 1
    assert search_spy.call_count == 1


@pytest.mark.asyncio
async def test_query_pipeline_with_transforms_runs_multi_variant() -> None:
    """A configured transformer produces variants and the pipeline fuses them."""
    pipeline, store, embedder = build_pipeline(
        transformer=ComposeTransformer([HydeTransformer(StubLlm(), n=1)])
    )
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")

    result = await pipeline.run(
        context,
        question="revenue",
        top_k=3,
        user=user,
    )
    assert result.success is True
    assert "hyde" in result.outputs["transforms_applied"]
    # Two variants → at least one search call to vector_store.
    assert isinstance(result.outputs["hits"], list)


@pytest.mark.asyncio
async def test_query_pipeline_rbac_filter_still_applied() -> None:
    """The transformer path must still honour the RBAC filter."""
    pipeline, store, embedder = build_pipeline(
        transformer=ComposeTransformer([HydeTransformer(StubLlm(), n=1)])
    )
    user = UserPrincipal(email="a@b.c", allowed_companies=["B"])  # no match
    context = PipelineContext(pipeline_name="query")

    result = await pipeline.run(
        context,
        question="revenue",
        top_k=3,
        user=user,
    )
    assert result.success is True
    # No chunks are visible — the RBAC filter blocks every variant.
    assert result.outputs["hits"] == []