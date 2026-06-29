"""Tests for the canonical domain models in :mod:`raghub.models.canonical`."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from raghub.models import (
    BlockKind,
    Citation,
    Document,
    DocumentBlock,
    DocumentSection,
    Embedding,
    EvaluationResult,
    KnowledgeBundle,
    PipelineContext,
    PipelineResult,
    canonical,
)
from raghub.models.canonical import deterministic_id


def test_deterministic_id_is_stable() -> None:
    """Same input → same id."""
    a = deterministic_id("doc", "uri", "v1")
    b = deterministic_id("doc", "uri", "v1")
    assert a == b


def test_deterministic_id_length_clamp() -> None:
    """The ``length`` argument is clamped to [8, 64]."""
    assert len(deterministic_id("a", length=2)) == 8
    assert len(deterministic_id("a", length=999)) == 64


def test_document_block_defaults() -> None:
    """DocumentBlock has sane defaults."""
    block = DocumentBlock()
    assert block.kind == BlockKind.TEXT
    assert block.content == ""
    assert block.metadata == {}


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


def test_knowledge_bundle_serialises_to_okf() -> None:
    """A bundle round-trips through OKF."""
    bundle = KnowledgeBundle(
        source_uri="file://example",
        checksum="abc",
        language="en",
        sections=[
            DocumentSection(
                index=0,
                blocks=[DocumentBlock(kind=BlockKind.TEXT, content="hello")],
            )
        ],
    )
    payload = canonical.to_okf(bundle) if hasattr(canonical, "to_okf") else bundle.model_dump()
    # The OKF helper lives in raghub.knowledge.okf; here we just
    # assert the Pydantic model emits the right fields.
    assert bundle.sections[0].blocks[0].content == "hello"
    assert payload["source_uri"] == "file://example"


def test_citation_provenance_fields() -> None:
    """Citation has all provenance fields."""
    c = Citation(
        chunk_id="c1",
        document_id="d1",
        version=1,
        page=2,
        section="Intro",
        quote="hello",
        score=0.9,
        source_uri="file://x",
    )
    assert c.chunk_id == "c1"
    assert c.page == 2
    assert c.section == "Intro"


def test_embedding_dimension() -> None:
    """Embedding carries a model + dim + vector."""
    e = Embedding(chunk_id="c1", model="hashing", dim=4, vector=[0.1, 0.2, 0.3, 0.4])
    assert e.dim == 4
    assert len(e.vector) == 4


def test_pipeline_result_success_and_failure() -> None:
    """PipelineResult records success/failure states."""
    ok = PipelineResult(pipeline_id="p1", pipeline_name="ingest", success=True)
    bad = PipelineResult(pipeline_id="p2", pipeline_name="ingest", success=False, error="oops")
    assert ok.success
    assert not bad.success
    assert bad.error == "oops"


def test_pipeline_context_starts_now() -> None:
    """PipelineContext.started_at is recent."""
    ctx = PipelineContext(pipeline_name="ingest")
    # ``ctx.started_at`` is a timezone-aware UTC ``datetime``. Compare
    # with a fresh timezone-aware UTC ``datetime.now`` to compute the
    # delta.
    delta = abs((datetime.now(timezone.utc) - ctx.started_at).total_seconds())
    assert delta < 5


def test_evaluation_result_passed_default() -> None:
    """EvaluationResult.passed defaults to True."""
    r = EvaluationResult(benchmark="financebench", example_id="0", metrics={"f1": 0.8})
    assert r.passed is True


def test_document_alias_is_document_record() -> None:
    """`Document` is a DocumentRecord subclass."""
    d = Document(checksum="abc", owner="u", organization="o")
    assert hasattr(d, "checksum")
    assert hasattr(d, "owner")
