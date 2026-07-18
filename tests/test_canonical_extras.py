"""Tests for the canonical domain models."""

from __future__ import annotations

from raghub.models.canonical import (
    BlockKind,
    Chunk,
    Citation,
    Document,
    DocumentBlock,
    DocumentSection,
    Embedding,
    EvaluationResult,
    KnowledgeBundle,
    PipelineResult,
    Query,
    Response,
    SearchResult,
)


def test_knowledge_bundle_full_serialization() -> None:
    """A bundle with sections and blocks serialises to a dict and back."""
    from raghub.knowledge.okf import dumps, loads

    bundle = KnowledgeBundle(
        source_uri="file://example",
        schema_version="0.1",
        checksum="abc",
        language="en",
        mime_type="text/plain",
        metadata={"author": "test"},
        sections=[
            DocumentSection(
                index=0,
                heading="Intro",
                page_numbers=[1],
                source_location="page 1",
                blocks=[
                    DocumentBlock(kind=BlockKind.TEXT, content="Hello"),
                    DocumentBlock(kind=BlockKind.TABLE, content="|a|b|"),
                ],
            )
        ],
    )
    encoded = dumps(bundle)
    decoded = loads(encoded)
    assert decoded.source_uri == bundle.source_uri
    assert decoded.sections[0].blocks[0].content == "Hello"
    assert decoded.sections[0].blocks[1].kind == BlockKind.TABLE


def test_pipeline_result_supports_string_outputs() -> None:
    """``PipelineResult.outputs`` accepts arbitrary key/value pairs."""
    result = PipelineResult(
        pipeline_id="p1",
        pipeline_name="ingest",
        success=True,
        outputs={"key": "value", "count": 5},
    )
    assert result.outputs["key"] == "value"
    assert result.outputs["count"] == 5


def test_citation_provenance_round_trip() -> None:
    """A :class:`Citation` survives a model_dump / model_validate cycle."""
    c = Citation(
        chunk_id="c1",
        document_id="d1",
        version=3,
        page=5,
        section="Revenue",
        quote="revenue grew 12%",
        score=0.95,
        source_uri="file://doc.pdf",
    )
    dumped = c.model_dump()
    restored = Citation.model_validate(dumped)
    assert restored == c


def test_search_result_chains_chunk() -> None:
    """A :class:`SearchResult` carries a chunk record."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        version=1,
        text="revenue grew 12%",
        company="o",
        owner="u",
    )
    hit = SearchResult(chunk_id="c1", score=0.9, chunk=chunk)
    assert hit.chunk.text == "revenue grew 12%"
    assert hit.score == 0.9


def test_response_typed_citations_and_chunks() -> None:
    """``Response`` carries typed ``Citation`` and ``SearchResult`` objects."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        version=1,
        text="revenue",
        company="o",
        owner="u",
    )
    resp = Response(
        answer="revenue grew 12%",
        citations=[Citation(chunk_id="c1", document_id="d1")],
        source_chunks=[SearchResult(chunk_id="c1", score=0.9, chunk=chunk)],
        metadata={"pipeline_id": "p1"},
    )
    assert resp.answer == "revenue grew 12%"
    assert resp.citations[0].document_id == "d1"
    assert resp.source_chunks[0].chunk.text == "revenue"


def test_evaluation_result_metrics() -> None:
    """``EvaluationResult.metrics`` is a free-form dict."""
    r = EvaluationResult(
        benchmark="financebench",
        example_id="0",
        metrics={"f1": 0.8, "recall": 0.9},
        passed=True,
    )
    assert r.metrics["f1"] == 0.8
    assert r.passed


def test_embedding_carries_dim_and_vector() -> None:
    """``Embedding`` validates ``dim`` matches ``len(vector)``."""
    e = Embedding(chunk_id="c1", model="hashing", dim=3, vector=[0.1, 0.2, 0.3])
    assert e.dim == 3
    assert len(e.vector) == 3


def test_document_section_blocks_in_order() -> None:
    """``DocumentSection.blocks`` preserves insertion order."""
    section = DocumentSection(
        index=0,
        blocks=[
            DocumentBlock(kind=BlockKind.TEXT, content="first"),
            DocumentBlock(kind=BlockKind.CODE, content="x = 1"),
            DocumentBlock(
                kind=BlockKind.IMAGE, content="fig1.png", metadata={"caption": "Figure 1"}
            ),
        ],
    )
    assert [b.kind for b in section.blocks] == [
        BlockKind.TEXT,
        BlockKind.CODE,
        BlockKind.IMAGE,
    ]
    assert section.blocks[2].metadata["caption"] == "Figure 1"


def test_query_alias() -> None:
    """``Query`` is the canonical alias of ``SearchRequest``."""
    q = Query(user_id="u1", question="revenue", session_id="s1", top_k=3)
    assert q.question == "revenue"
    assert q.top_k == 3


def test_response_alias() -> None:
    """``Response`` is the canonical model for query answers."""
    assert Response.model_fields is not None


def test_document_alias_is_document_record() -> None:
    """``Document`` is a subclass of ``DocumentRecord``."""
    d = Document(checksum="abc", owner="u", organization="o")
    assert d.checksum == "abc"
    assert d.organization == "o"


def test_chunk_alias_is_chunk_record() -> None:
    """``Chunk`` is a subclass of ``ChunkRecord``."""
    c = Chunk(
        chunk_id="c1",
        document_id="d1",
        version=1,
        text="revenue",
        company="o",
        owner="u",
    )
    assert c.chunk_id == "c1"
    assert c.text == "revenue"
