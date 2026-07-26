"""Phase 6.2 — RAPTOR index tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from raghub.embeddings import HashingEmbeddingProvider
from raghub.knowledge import RaptorIndex
from raghub.models import Chunk


class StubLlm:
    """LLM stub that echoes a summary built from the input's first 80 chars."""

    def __init__(self) -> None:
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
        # Extract the passage body from the prompt and summarise.
        body = question.split("Passage:", 1)[-1].strip()
        words = body.split()
        return f"summary of: {' '.join(words[:8])}"


@pytest.fixture
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dimension=16, model_name="t")


def make_chunks(texts: list[str], prefix: str = "c") -> list[Chunk]:
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"{prefix}-{i}",
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
        )
    return chunks


def test_raptor_initialises_with_defaults() -> None:
    index = RaptorIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8))
    assert index.name == "raptor"
    assert index.health() == {"name": "raptor", "levels": {}, "lock_token": 0}


def test_raptor_add_chunks_creates_leaf_level() -> None:
    embedder = HashingEmbeddingProvider(dimension=16)
    index = RaptorIndex(llm=StubLlm(), embedder=embedder, depth=1, cluster_size=2)
    chunks = make_chunks(
        [
            "Revenue grew twelve percent in Q3 driven by SaaS bookings",
            "Operating margin expanded two hundred basis points sequentially",
            "Customer count rose eight percent year over year",
            "Free cash flow conversion improved materially in Q3",
        ]
    )
    vectors = [embedder.embed_text(c.text) for c in chunks]
    index.add_chunks(chunks, vectors)
    health = index.health()
    assert health["levels"]["level_0"] == 4
    # depth=1 → exactly one summary level above the leaves.
    assert health["levels"]["level_1"] >= 1

    # Verify the upper-level summaries are real (non-empty, not
    # the original text), and that they were created by the LLM.
    summaries = index.levels[1]
    assert summaries, "no summaries built at level 1"
    for summary in summaries:
        assert summary.text, "summary text is empty"
        assert summary.text != chunks[0].text, "summary must not equal leaf text"
        assert summary.metadata.get("raptor_level") == 1
        # The LLM stub returned "summary of: ...".
        assert summary.text.startswith("summary of:"), summary.text


def test_raptor_search_uses_all_levels() -> None:
    """A query that matches a summary's text surfaces the summary hit.

    We seed with depth=0 (leaf only) plus a forced level-1 entry
    by calling rebuild_tree with a stub LLM, then search for a
    phrase that appears only in the summary — proves the index
    searches both levels, not just the leaves.
    """
    embedder = HashingEmbeddingProvider(dimension=16)
    index = RaptorIndex(llm=StubLlm(), embedder=embedder, depth=0, cluster_size=2)
    chunks = make_chunks(
        [
            "the magenta widget is large",
            "the cyan widget is small",
        ]
    )
    vectors = [embedder.embed_text(c.text) for c in chunks]
    index.add_chunks(chunks, vectors)
    # No summaries were built (depth=0). Build a single fake
    # summary level manually and add a chunk with a unique token
    # so we can prove search finds it.
    from raghub.models import ChunkRecord

    summary_chunk = ChunkRecord(
        chunk_id="sum-unique",
        document_id="d",
        version=1,
        page=1,
        source_location="raptor://summary",
        section="",
        company="A",
        owner="",
        department="",
        text="a paragraph that talks about blueberry exclusively",
        metadata={"vector": embedder.embed_text("blueberry"), "raptor_level": 1},
    )
    index.levels.append([summary_chunk])
    hits = index.search("blueberry", top_k=5)
    assert any(h.chunk.chunk_id == "sum-unique" for h in hits), hits


def test_raptor_search_returns_hits_with_cosine_score() -> None:
    embedder = HashingEmbeddingProvider(dimension=16)
    index = RaptorIndex(llm=StubLlm(), embedder=embedder, depth=1, cluster_size=2)
    chunks = make_chunks(
        [
            "Revenue grew twelve percent in Q3 driven by SaaS bookings",
            "Operating margin expanded two hundred basis points sequentially",
            "Customer count rose eight percent year over year",
            "Free cash flow conversion improved materially in Q3",
        ]
    )
    vectors = [embedder.embed_text(c.text) for c in chunks]
    index.add_chunks(chunks, vectors)
    hits = index.search("revenue growth Q3", top_k=3)
    assert len(hits) >= 1
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_raptor_delete_for_document_purges_leaves() -> None:
    embedder = HashingEmbeddingProvider(dimension=16)
    index = RaptorIndex(llm=StubLlm(), embedder=embedder, depth=0, cluster_size=2)
    chunks = make_chunks(["alpha", "beta", "gamma"])
    vectors = [embedder.embed_text(c.text) for c in chunks]
    index.add_chunks(chunks, vectors)
    removed = index.delete_for_document("d")
    assert removed == 3
    assert index.search("alpha", top_k=3) == []


def test_raptor_search_on_empty_index_returns_empty() -> None:
    index = RaptorIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8))
    assert index.search("anything", top_k=3) == []


def test_raptor_works_without_llm_when_depth_zero() -> None:
    """A depth=0 RAPTOR is just a leaf-level cosine search."""
    embedder = HashingEmbeddingProvider(dimension=16)
    index = RaptorIndex(llm=None, embedder=embedder, depth=0, cluster_size=5)
    chunks = make_chunks(
        [
            "Revenue grew twelve percent in Q3 driven by SaaS bookings",
            "Operating margin expanded two hundred basis points sequentially",
            "Customer count rose eight percent year over year",
        ]
    )
    vectors = [embedder.embed_text(c.text) for c in chunks]
    index.add_chunks(chunks, vectors)
    # No summary levels were built.
    assert len(index.health()["levels"]) == 1
    hits = index.search("margin expansion", top_k=2)
    assert len(hits) >= 1


def test_raptor_add_chunks_rejects_mismatched_lengths() -> None:
    index = RaptorIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8))
    chunks = make_chunks(["a", "b"])
    with pytest.raises(ValueError):
        index.add_chunks(chunks, [[0.1, 0.2]])  # only one vector


def test_raptor_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        RaptorIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8), depth=-1)
    with pytest.raises(ValueError):
        RaptorIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8), cluster_size=0)