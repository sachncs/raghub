"""Phase 6.3 — GraphRAG index tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from raghub.embeddings import HashingEmbeddingProvider
from raghub.knowledge import GraphRagIndex
from raghub.models import Chunk


class StubLlm:
    """LLM stub that produces canned extraction / summarisation payloads.

    The stub is context-aware: it returns only the entities
    mentioned in the chunk text it received. This avoids
    cross-contamination in tests that seed multiple chunks with
    different topics.
    """

    def __init__(
        self,
        *,
        entities: list[dict[str, str]] | None = None,
        triples: list[dict[str, str]] | None = None,
        summaries: list[str] | None = None,
    ) -> None:
        self._entities = entities or []
        self._triples = triples or []
        self._summaries = summaries or []
        self.call_idx = 0

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
        self.call_idx += 1
        if "Entities:" in question:
            # Summarise-community path; round-robin through the canned summaries.
            if not self._summaries:
                return "community summary"
            return self._summaries[(self.call_idx - 1) % len(self._summaries)]
        # Extract path: only return entities that are mentioned in
        # the chunk text (the EXTRACT_PROMPT puts the passage in the
        # question body). This prevents cross-contamination when a
        # single stub serves multiple chunks with different topics.
        chunk_entities = [
            e for e in self._entities
            if e.get("name") and e["name"] in question
        ]
        chunk_triples = [
            t for t in self._triples
            if t.get("subject") in question and t.get("object") in question
        ]
        return json.dumps({"entities": chunk_entities, "triples": chunk_triples})


def make_chunk(text: str, cid: str = "c") -> Chunk:
    return Chunk(
        chunk_id=cid,
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


def build_index(
    *,
    entities: list[dict[str, str]],
    triples: list[dict[str, str]],
    summaries: list[str] | None = None,
    texts: list[str] | None = None,
) -> GraphRagIndex:
    embedder = HashingEmbeddingProvider(dimension=16)
    llm = StubLlm(entities=entities, triples=triples, summaries=summaries)
    index = GraphRagIndex(llm=llm, embedder=embedder)
    if texts:
        chunks = [make_chunk(t, cid=f"c-{i}") for i, t in enumerate(texts)]
        vectors = [embedder.embed_text(c.text) for c in chunks]
        index.add_chunks(chunks, vectors)
    return index


def test_graphrag_add_chunks_extracts_entities_and_triples() -> None:
    idx = build_index(
        entities=[{"name": "Acme", "type": "Company"}, {"name": "Q3", "type": "Date"}],
        triples=[{"subject": "Acme", "predicate": "reports", "object": "Q3"}],
        texts=[
            "Acme reported Q3 results to shareholders",
            "Q3 revenue grew twelve percent",
        ],
    )
    # The graph connected the two entities.
    assert "Acme" in idx.graph["Q3"]
    assert "Q3" in idx.graph["Acme"]


def test_graphrag_health_reports_chunks() -> None:
    idx = build_index(
        entities=[{"name": "Acme", "type": "Company"}],
        triples=[],
        texts=["Acme text"],
    )
    assert idx.health() == {"name": "graphrag", "chunks": 1}


def test_graphrag_search_local_returns_entity_anchored_hits() -> None:
    """``search_local`` returns chunks whose entity overlaps with the query.

    We seed a chunk mentioning ``Acme`` and a chunk mentioning an
    unrelated entity; only the Acme chunk should surface when the
    query mentions ``Acme``.
    """
    embedder = HashingEmbeddingProvider(dimension=16)
    from raghub.knowledge import GraphRagIndex

    llm = StubLlm(
        entities=[
            {"name": "Acme", "type": "Company"},
            {"name": "Globex", "type": "Company"},
        ],
        triples=[],
    )
    index = GraphRagIndex(llm=llm, embedder=embedder, hop_limit=1)
    chunks = [
        make_chunk("Acme reported strong results in Q3 with growth", "c-acme"),
        make_chunk("Globex reported weak results in Q3 with decline", "c-globex"),
    ]
    vectors = [embedder.embed_text(c.text) for c in chunks]
    index.add_chunks(chunks, vectors)
    hits = index.search_local("What did Acme report?", top_k=5)
    # The Acme chunk must surface.
    hit_ids = [h.chunk.chunk_id for h in hits]
    assert "c-acme" in hit_ids
    # The Globex chunk must NOT (no Acme in its entities).
    assert "c-globex" not in hit_ids
    # And the Globex chunk should appear in a separate query.
    globex_hits = index.search_local("What did Globex report?", top_k=5)
    globex_ids = [h.chunk.chunk_id for h in globex_hits]
    assert "c-globex" in globex_ids
    assert "c-acme" not in globex_ids


def test_graphrag_search_global_returns_community_summaries() -> None:
    """``search_global`` surfaces community summaries ordered by query overlap.

    The canned summary is ``"Acme reported strong results"``; the
    query is constructed so it tokenises into a superset of the
    summary's tokens, giving a positive overlap score.
    """
    from raghub.knowledge import GraphRagIndex

    embedder = HashingEmbeddingProvider(dimension=16)
    llm = StubLlm(
        entities=[{"name": "Acme", "type": "Company"}],
        triples=[],
        summaries=["Acme reported strong results"],
    )
    index = GraphRagIndex(llm=llm, embedder=embedder, hop_limit=1)
    index.add_chunks(
        [make_chunk("Acme reported strong results in Q3", "c-0")],
        [embedder.embed_text("Acme reported strong results in Q3")],
    )
    hits = index.search_global(
        "What were the strong results that Acme reported?",
        top_k=5,
    )
    assert hits, "search_global returned no hits"
    top = hits[0]
    # The community summary itself is the chunk text.
    assert "Acme reported strong results" in top.chunk.text
    assert top.chunk.metadata.get("graphrag_community") is True
    # The summary's score must be positive.
    assert top.score > 0


def test_graphrag_combined_search_dedupes() -> None:
    idx = build_index(
        entities=[{"name": "Acme", "type": "Company"}],
        triples=[],
        summaries=["Acme summary"],
        texts=["Acme is great"],
    )
    out = idx.search("Acme", top_k=5)
    chunk_ids = [h.chunk.chunk_id for h in out]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_graphrag_combined_search_runs_local_and_global() -> None:
    """``search`` actually runs both channels and dedupes their output.

    We seed the index with a single isolated entity and verify that
    a query mentioning the entity returns BOTH the chunk hit
    (local channel) and the community summary (global channel) —
    the same chunk_id never appears twice.
    """
    from raghub.knowledge import GraphRagIndex
    from raghub.models import Chunk

    embedder = HashingEmbeddingProvider(dimension=16)
    index = GraphRagIndex(llm=None, embedder=embedder, hop_limit=1)
    chunk = Chunk(
        chunk_id="doc-1",
        document_id="d",
        version=1,
        page=1,
        source_location="s",
        section="",
        company="A",
        owner="",
        department="",
        text="Acme reported strong results in Q3",
        metadata={"vector": embedder.embed_text("Acme reported strong results in Q3")},
    )
    index.chunks["doc-1"] = chunk
    index.graph["Acme"] = set()
    index.entity_chunks["Acme"] = {"doc-1"}
    index.chunk_entities["doc-1"] = {"Acme"}
    index.communities = [{"Acme"}]
    index.community_summaries[0] = "Acme reported strong results"

    out = index.search("Acme reported", top_k=5)
    # Both channels produced hits.
    local_hits = [h for h in out if not h.chunk.metadata.get("graphrag_community")]
    global_hits = [h for h in out if h.chunk.metadata.get("graphrag_community")]
    assert any(h.chunk_id == "doc-1" for h in local_hits), local_hits
    assert any("Acme" in h.chunk.text for h in global_hits), global_hits
    # And the dedup invariant holds.
    assert len({h.chunk_id for h in out}) == len(out)


def test_graphrag_delete_for_document_removes_entities() -> None:
    idx = build_index(
        entities=[{"name": "Acme", "type": "Company"}],
        triples=[],
        texts=["Acme text"],
    )
    removed = idx.delete_for_document("d")
    assert removed >= 1
    assert idx.search_local("Acme", top_k=5) == []


def test_graphrag_search_empty_index_returns_empty() -> None:
    idx = GraphRagIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8))
    assert idx.search("anything", top_k=5) == []
    assert idx.search_local("anything", top_k=5) == []
    assert idx.search_global("anything", top_k=5) == []


def test_graphrag_unknown_mode_in_tool_returns_error() -> None:
    """The graph_search tool surfaces an unknown mode as a ToolResult error."""
    from raghub.tools.graph_search import GraphSearchTool

    tool = GraphSearchTool(GraphRagIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8)))
    import asyncio

    result = asyncio.run(tool.execute(None, query="q", mode="weird", top_k=5))  # type: ignore[arg-type]
    assert result.ok is False
    assert "unknown mode" in result.error


def test_graphrag_constructor_validates_hop_limit() -> None:
    with pytest.raises(ValueError):
        GraphRagIndex(llm=StubLlm(), embedder=HashingEmbeddingProvider(dimension=8), hop_limit=-1)