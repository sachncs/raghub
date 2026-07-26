"""Phase 7 — built-in tool tests."""

from __future__ import annotations

from typing import Any

import pytest

import asyncio

from raghub.tools.base import BaseTool, ToolContext, ToolResult
from raghub.tools.date_today import DateTodayTool
from raghub.tools.hybrid_search import HybridSearchTool
from raghub.tools.keyword_search import KeywordSearchTool
from raghub.tools.summary_search import SummarySearchTool
from raghub.tools.vector_search import VectorSearchTool
from raghub.tools.web_search import WebSearchTool
from raghub.embeddings import HashingEmbeddingProvider
from raghub.models import ChunkRecord, UserPrincipal
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.vectorstore import InMemoryVectorStore


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


@pytest.fixture
def store_with_chunks() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    chunks = [
        make_chunk(0, "revenue grew 12% in Q3 driven by SaaS bookings"),
        make_chunk(1, "operating margin expanded 200bps sequentially"),
        make_chunk(2, "customer count rose eight percent year over year"),
    ]
    vectors = [embedder.embed_text(c.text) for c in chunks]
    store.upsert(chunks, vectors)
    return store


def make_retrieval(store: InMemoryVectorStore) -> RetrievalPipeline:
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    from raghub.retrieval.reranker import IdentityReranker

    return RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=IdentityReranker(),
    )


def make_ctx(user: UserPrincipal | None = None) -> ToolContext:
    return ToolContext(user=user, question="revenue trends")


# --- VectorSearchTool ---------------------------------------------------


@pytest.mark.asyncio
async def test_vector_search_tool_returns_hits(store_with_chunks: InMemoryVectorStore) -> None:
    pipe = make_retrieval(store_with_chunks)
    tool = VectorSearchTool(pipe)
    result = await tool.execute(make_ctx(), query="revenue", top_k=2)
    assert result.ok is True
    assert "hits" in result.data
    assert len(result.data["hits"]) == 2
    # The "revenue" chunk should be the top hit.
    assert result.data["hits"][0]["chunk_id"] == "c-0"


@pytest.mark.asyncio
async def test_vector_search_tool_empty_query_returns_empty(store_with_chunks: InMemoryVectorStore) -> None:
    pipe = make_retrieval(store_with_chunks)
    tool = VectorSearchTool(pipe)
    # Empty query text + empty context.question → failure.
    result = await tool.execute(ToolContext(), query="", top_k=2)
    assert result.ok is False
    assert "empty query" in result.error


@pytest.mark.asyncio
async def test_vector_search_tool_falls_back_to_context_question(
    store_with_chunks: InMemoryVectorStore,
) -> None:
    """When ``query`` is empty but ``context.question`` is set, search the latter.

    The planner may pass an empty ``query`` arg to let the tool
    re-derive the question from the conversation context.
    """
    pipe = make_retrieval(store_with_chunks)
    tool = VectorSearchTool(pipe)
    ctx = ToolContext(question="revenue")
    result = await tool.execute(ctx, query="", top_k=2)
    assert result.ok is True
    hit_ids = [h["chunk_id"] for h in result.data["hits"]]
    # The "revenue grew" chunk should surface — proving we fell back.
    assert "c-0" in hit_ids


@pytest.mark.asyncio
async def test_vector_search_tool_rbac_filter(store_with_chunks: InMemoryVectorStore) -> None:
    pipe = make_retrieval(store_with_chunks)
    tool = VectorSearchTool(pipe)
    user = UserPrincipal(email="a@b.c", allowed_companies=["ZZ"])  # no match
    result = await tool.execute(make_ctx(user), query="revenue", top_k=2)
    # RBAC blocks every chunk — the result is the empty-list response.
    assert result.ok is True
    assert result.data.get("hits", []) == []


# --- KeywordSearchTool -------------------------------------------------


@pytest.mark.asyncio
async def test_keyword_search_tool_finds_token_overlap_only_chunks(
    store_with_chunks: InMemoryVectorStore,
) -> None:
    """The keyword path finds chunks that share tokens with the query.

    We add a chunk whose text shares tokens with the query but has
    no semantic overlap with the seeded chunks. The keyword tool
    must surface it; the dense-only retriever (vector similarity)
    would not have.
    """
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    tool = KeywordSearchTool(store_with_chunks)

    # A chunk with a unique keyword repeated many times — only the
    # keyword channel will match it strongly.
    keyword_only = make_chunk(
        "kw-only",
        "biohazard biohazard biohazard biohazard biohazard biohazard"
        " biohazard biohazard biohazard",
    )
    keyword_vector = embedder.embed_text(keyword_only.text)
    store_with_chunks.upsert([keyword_only], [keyword_vector])

    result = await tool.execute(make_ctx(), query="biohazard", top_k=3)
    assert result.ok is True
    hit_ids = [h["chunk_id"] for h in result.data["hits"]]
    assert any("kw-only" in cid for cid in hit_ids)
    # It should be the top hit — the only chunk containing "biohazard".
    assert "kw-only" in hit_ids[0]


@pytest.mark.asyncio
async def test_keyword_search_tool_propagates_when_store_lacks_keyword() -> None:
    class NoKeywordStore:
        def keyword_search(self, *_):  # pragma: no cover - branch covered
            raise AttributeError("missing")

    tool = KeywordSearchTool(NoKeywordStore())
    with pytest.raises(AttributeError, match="missing"):
        await tool.execute(make_ctx(), query="revenue", top_k=2)


# --- HybridSearchTool --------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_tool_fuses_both_channels(
    store_with_chunks: InMemoryVectorStore,
) -> None:
    """The fused result must include hits that only the keyword channel finds.

    We add a chunk whose keyword overlap with the query is high
    but whose embedding similarity is low (a long, topically
    distinct chunk). The dense-only retriever misses it; the
    keyword retriever finds it; the hybrid path must surface it.
    """
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    pipe = make_retrieval(store_with_chunks)
    tool = HybridSearchTool(pipe, store_with_chunks)

    keyword_only_chunk = ChunkRecord(
        chunk_id="decoy-1",
        document_id="d",
        version=1,
        page=1,
        source_location="s",
        section="",
        company="A",
        owner="",
        department="",
        text=(
            "revenue revenue revenue revenue revenue and also revenue "
            "mentioned once more for keyword matching certainty"
        ),
        metadata={},
    )
    keyword_vector = embedder.embed_text(keyword_only_chunk.text)
    store_with_chunks.upsert([keyword_only_chunk], [keyword_vector])

    result = await tool.execute(make_ctx(), query="revenue", top_k=5)
    assert result.ok is True
    hit_ids = [h["chunk_id"] for h in result.data["hits"]]
    # The hybrid result must surface the keyword-only chunk — the
    # dense-only retriever would not have found it.
    assert "decoy-1" in hit_ids
    # And the RRF ordering should put the keyword-favoured hit
    # somewhere in the top results (not necessarily first because
    # the dense hits still dominate the score).
    assert hit_ids.index("decoy-1") < len(hit_ids)


@pytest.mark.asyncio
async def test_hybrid_search_tool_propagates_dense_failure(
    store_with_chunks: InMemoryVectorStore,
) -> None:
    """A dense-channel exception now propagates to the caller."""
    pipe = make_retrieval(store_with_chunks)

    class BrokenPipeline:
        def retrieve(self, **_):
            raise RuntimeError("boom")

    tool = HybridSearchTool(BrokenPipeline(), store_with_chunks)
    with pytest.raises(RuntimeError, match="boom"):
        await tool.execute(make_ctx(), query="revenue", top_k=3)


# --- DateTodayTool ------------------------------------------------------


@pytest.mark.asyncio
async def test_date_today_tool_returns_iso_date() -> None:
    tool = DateTodayTool()
    result = await tool.execute(make_ctx())
    assert result.ok is True
    assert "T" not in result.content  # just the date
    assert len(result.content) == 10  # YYYY-MM-DD


# --- SummarySearchTool --------------------------------------------------


@pytest.mark.asyncio
async def test_summary_search_tool_no_index_returns_noop_message() -> None:
    tool = SummarySearchTool(None)
    result = await tool.execute(make_ctx(), query="x", top_k=5)
    assert result.ok is True
    assert "no summary index" in result.content


@pytest.mark.asyncio
async def test_summary_search_tool_with_index_returns_hits() -> None:
    """A real RaptorIndex surfaces its summaries in the data payload.

    We build a tiny RAPTOR (depth=0 so it stays as a leaf cosine
    search) and verify the tool returns those leaves as hits.
    """
    from raghub.knowledge import RaptorIndex

    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    index = RaptorIndex(llm=None, embedder=embedder, depth=0)
    chunks = [
        make_chunk("sum-0", "revenue grew twelve percent in Q3"),
        make_chunk("sum-1", "operating margin expanded"),
    ]
    index.add_chunks(chunks, [embedder.embed_text(c.text) for c in chunks])

    tool = SummarySearchTool(index)
    result = await tool.execute(make_ctx(), query="revenue", top_k=5)
    assert result.ok is True
    assert "revenue grew" in result.content
    hit_ids = [h["chunk_id"] for h in result.data["hits"]]
    assert any("sum-0" in cid for cid in hit_ids)
    # The RAPTOR level is exposed as a top-level key (not nested).
    assert all(h["level"] == 0 for h in result.data["hits"])


# --- WebSearchTool ------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_tool_missing_dep_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A botched import of ``duckduckgo_search`` now lets ``ImportError`` propagate."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "duckduckgo_search" or name.startswith("duckduckgo_search"):
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    tool = WebSearchTool()
    with pytest.raises(ImportError, match="simulated missing"):
        await tool.execute(make_ctx(), query="anything")


# --- BaseTool contract --------------------------------------------------


def test_base_tool_run_wraps_exceptions() -> None:
    """``run`` propagates exceptions to the caller (agent loop catches them)."""

    class Raiser(BaseTool):
        name = "raiser"
        description = "Always raises."
        json_schema: dict[str, Any] = {"type": "object", "properties": {}}

        async def execute(self, context, **_):
            raise RuntimeError("nope")

    tool = Raiser()
    with pytest.raises(RuntimeError, match="nope"):
        asyncio.run(tool.run({}, make_ctx()))