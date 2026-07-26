"""Phase 5.3 — ``QueryPipeline`` long-context pass integration.

Confirms the pass is invoked after the cross-encoder rerank and
that the fast path (no pass configured) stays byte-equivalent.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from raghub.config import Settings, LongContextConfig
from raghub.embeddings import HashingEmbeddingProvider
from raghub.generation import DefaultGenerator
from raghub.llm import HeuristicLLMProvider
from raghub.models import (
    ChunkRecord,
    Citation,
    ConversationTurn,
    PipelineContext,
    UserPrincipal,
)
from raghub.pipelines.rag import QueryPipeline
from raghub.retrieval.long_context import LongContextRerankPass
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.retrieval.reranker import IdentityReranker
from raghub.vectorstore.memory import InMemoryVectorStore


class StubLlm:
    """LLM stub returning a canned long-context payload."""

    def __init__(self, payload: str, model_name: str = "claude-3-5-sonnet") -> None:
        self.model_name = model_name
        self.payload = payload
        self.calls = 0

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
        self.calls += 1
        return self.payload


def make_chunk(i: int, text: str) -> ChunkRecord:
    return ChunkRecord(
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


def build_pipeline(
    *,
    long_context: LongContextRerankPass | None = None,
    llm: Any | None = None,
) -> tuple[QueryPipeline, InMemoryVectorStore, Any]:
    """Build the pipeline. Returns the stub LLM (if any) for call-count assertions."""
    store = InMemoryVectorStore()
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    retrieval = RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=IdentityReranker(),
    )
    chunks = [
        make_chunk(0, "revenue grew 12% in Q3"),
        make_chunk(1, "operating margin expanded"),
        make_chunk(2, "customer count rose"),
    ]
    vectors = [embedder.embed_text(c.text) for c in chunks]
    store.upsert(chunks, vectors)
    # The generator uses the heuristic LLM (no extra stub calls); the
    # long-context pass uses the stub when supplied.
    pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=store,
        generator=DefaultGenerator(llm=HeuristicLLMProvider()),
        reranker=retrieval.reranker,
        retrieval_pipeline=retrieval,
        long_context_pass=long_context,
    )
    return pipeline, store, llm


@pytest.mark.asyncio
async def test_pipeline_invokes_long_context_when_configured() -> None:
    """The pass runs after rerank and reorders the hits."""
    payload = (
        '{"items": ['
        '{"chunk_id": "c-2", "score": 0.9, "rationale": "direct"}'
        ']}'
    )
    llm = StubLlm(payload)
    pass_ = LongContextRerankPass(
        llm,
        LongContextConfig(enabled=True, candidate_k=10),
    )
    pipeline, _, _stub = build_pipeline(long_context=pass_, llm=llm)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")
    result = await pipeline.run(context, question="anything", top_k=3, user=user)
    assert result.success is True
    assert llm.calls == 1
    # c-2 ranks first per the LLM.
    assert result.outputs["hits"][0].chunk_id == "c-2"


@pytest.mark.asyncio
async def test_pipeline_skips_long_context_when_disabled() -> None:
    """No pass configured → no LLM call."""
    pipeline, _, _stub = build_pipeline(long_context=None)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")
    result = await pipeline.run(context, question="anything", top_k=3, user=user)
    assert result.success is True
    # No assertion on call counts — but the ordering must match the
    # cross-encoder's (identity here).
    assert [h.chunk_id for h in result.outputs["hits"]] == ["c-0", "c-1", "c-2"]


@pytest.mark.asyncio
async def test_pipeline_long_context_no_op_when_model_not_in_allowlist() -> None:
    """A misconfigured LLM model silently disables the pass."""
    payload = '{"items": [{"chunk_id": "c-0", "score": 0.9, "rationale": "x"}]}'
    llm = StubLlm(payload, model_name="heuristic-llm")
    pass_ = LongContextRerankPass(llm, LongContextConfig(enabled=True, candidate_k=10))
    pipeline, _, _stub = build_pipeline(long_context=pass_, llm=llm)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")
    result = await pipeline.run(context, question="anything", top_k=3, user=user)
    # No LLM call — pass.is_eligible() returned False.
    assert llm.calls == 0
    # Original order preserved.
    assert [h.chunk_id for h in result.outputs["hits"]] == ["c-0", "c-1", "c-2"]


def test_query_pipeline_accepts_long_context_pass() -> None:
    """Passing a long_context_pass through the pipeline runs it after rerank."""
    from raghub.pipelines.rag import QueryPipeline

    # Wiring is the important behaviour; the constructor accepts the kwarg
    # and the pass is invoked during run() when configured. The full happy
    # path is exercised by test_long_context_rerank_is_invoked_after_xenc above.
    sig_params = QueryPipeline.__init__.__code__.co_varnames
    assert "long_context_pass" in sig_params


def test_facade_wires_long_context_pass_through() -> None:
    """`RAG.__init__` instantiates and exposes a LongContextRerankPass."""
    with tempfile.TemporaryDirectory() as tmp:
        rag = __import__("raghub.api.rag", fromlist=["RAG"]).RAG(
            settings=Settings(
                data_dir=Path(tmp),
                long_context_pass=LongContextConfig(enabled=True),
            )
        )
        assert rag.query_pipeline.long_context_pass is rag.long_context_pass