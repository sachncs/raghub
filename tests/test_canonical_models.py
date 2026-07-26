"""Tests for the canonical domain models in :mod:`raghub.models.canonical`."""

from __future__ import annotations

from datetime import UTC, datetime

from raghub.models import (
    BlockKind,
    Citation,
    Chunk,
    Document,
    DocumentBlock,
    DocumentSection,
    Embedding,
    EvaluationResult,
    KnowledgeBundle,
    PipelineContext,
    PipelineResult,
    Query,
    Response,
    SearchResult,
    deterministic_id,
)


def test_deterministic_id_is_stable() -> None:
    """Same input produces same id."""
    a = deterministic_id("doc", "uri", "v1")
    b = deterministic_id("doc", "uri", "v1")
    assert a == b


def test_deterministic_id_length_clamp() -> None:
    """The length argument is clamped to [8, 64]."""
    assert len(deterministic_id("a", length=2)) == 8
    assert len(deterministic_id("a", length=999)) == 64


def test_document_block_defaults_round_trip() -> None:
    """DocumentBlock defaults survive serialization round-trip."""
    block = DocumentBlock()
    dumped = block.model_dump()
    assert dumped["kind"] == "text"
    assert dumped["content"] == ""
    assert dumped["metadata"] == {}
    restored = DocumentBlock.model_validate(dumped)
    assert restored == block


def test_document_section_round_trip() -> None:
    """A section carries blocks in order."""
    section = DocumentSection(
        index=2,
        heading="Revenue",
        page_numbers=[3, 4],
        source_location="page 3",
        blocks=[DocumentBlock(kind=BlockKind.TEXT, content="Hello")],
    )
    assert section.blocks[0].content == "Hello"
    assert section.index == 2


def test_document_section_blocks_preserve_kind() -> None:
    """DocumentSection.blocks preserves insertion order and block kinds."""
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


def test_knowledge_bundle_full_okf_round_trip() -> None:
    """A bundle with sections and blocks survives OKF dumps/loads."""
    from raghub.knowledge import dumps, loads

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


def test_citation_provenance_round_trip() -> None:
    """A Citation survives a model_dump / model_validate cycle."""
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


def test_embedding_round_trip() -> None:
    """Embedding survives serialization round-trip."""
    e = Embedding(chunk_id="c1", model="hashing", dim=3, vector=[0.1, 0.2, 0.3])
    dumped = e.model_dump()
    restored = Embedding.model_validate(dumped)
    assert restored.dim == 3
    assert restored.vector == [0.1, 0.2, 0.3]
    assert restored.chunk_id == "c1"


def test_pipeline_result_success_and_failure() -> None:
    """PipelineResult records success/failure states."""
    ok = PipelineResult(pipeline_id="p1", pipeline_name="ingest", success=True)
    bad = PipelineResult(pipeline_id="p2", pipeline_name="ingest", success=False, error="oops")
    assert ok.success
    assert not bad.success
    assert bad.error == "oops"


def test_pipeline_result_supports_arbitrary_outputs() -> None:
    """PipelineResult.outputs accepts arbitrary key/value pairs."""
    result = PipelineResult(
        pipeline_id="p1",
        pipeline_name="ingest",
        success=True,
        outputs={"key": "value", "count": 5},
    )
    assert result.outputs["key"] == "value"
    assert result.outputs["count"] == 5


def test_pipeline_context_starts_now() -> None:
    """PipelineContext.started_at is recent."""
    ctx = PipelineContext(pipeline_name="ingest")
    delta = abs((datetime.now(UTC) - ctx.started_at).total_seconds())
    assert delta < 5


def test_evaluation_result_passed_default() -> None:
    """EvaluationResult.passed defaults to True."""
    r = EvaluationResult(benchmark="financebench", example_id="0", metrics={"f1": 0.8})
    assert r.passed is True


def test_evaluation_result_metrics() -> None:
    """EvaluationResult.metrics is a free-form dict."""
    r = EvaluationResult(
        benchmark="financebench",
        example_id="0",
        metrics={"f1": 0.8, "recall": 0.9},
        passed=True,
    )
    assert r.metrics["f1"] == 0.8
    assert r.passed


def test_search_result_chains_chunk() -> None:
    """A SearchResult carries a chunk record with score."""
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
    """Response carries typed Citation and SearchResult objects."""
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


def test_query_alias() -> None:
    """Query is the canonical alias of SearchRequest."""
    q = Query(user_id="u1", question="revenue", session_id="s1", top_k=3)
    assert q.question == "revenue"
    assert q.top_k == 3


def test_document_alias() -> None:
    """Document carries checksum, owner, and organization."""
    d = Document(checksum="abc", owner="u", organization="o")
    assert d.checksum == "abc"
    assert d.organization == "o"


def test_chunk_alias() -> None:
    """Chunk carries chunk_id, text, company, and owner."""
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
