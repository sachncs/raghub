"""Tests for entity verify() methods.

Each test exercises the public ``verify()`` method on the model
classes that Phase 1.7 introduced. Construction is permissive
(Pydantic only enforces field types); invariant violations
surface through ``verify()`` which raises :class:`VerificationError`.
"""

from __future__ import annotations

import pytest

from raghub.errors import VerificationError
from raghub.models import (
    ChunkRecord,
    Citation,
    Hit,
    PipelineResult,
    Response,
)


def test_chunkverify_requires_checksum():
    """verify() raises on a chunk with a checksum that doesn't match its text."""
    c = ChunkRecord(
        chunk_id="x",
        document_id="d",
        version=1,
        text="hello",
        company="",
        owner="",
        checksum="bogus",
    )
    with pytest.raises(VerificationError, match="checksum"):
        c.verify()


def test_retrieval_hit_requires_matching_chunk_id(sample_chunk):
    """verify() raises when Hit.chunk_id and chunk.chunk_id disagree."""
    hit = Hit(chunk_id="wrong-id", score=0.5, chunk=sample_chunk)
    with pytest.raises(VerificationError, match="does not match"):
        hit.verify()


def test_response_citation_must_match_source_chunk(sample_chunk):
    """verify() raises when a citation's chunk_id is missing from source_chunks."""
    sample_chunk.chunk_id = "abc"
    citation = Citation(chunk_id="xyz", document_id="d", version=1)
    response = Response(
        answer="x",
        citations=[citation],
        source_chunks=[Hit(chunk_id="abc", score=0.5, chunk=sample_chunk)],
    )
    with pytest.raises(VerificationError, match="not in source_chunks"):
        response.verify()


def test_pipelineresult_success_no_error():
    """A successful PipelineResult verifies."""
    r = PipelineResult(pipeline_id="a", pipeline_name="b", success=True)
    r.verify()


def test_pipelineresult_failure_requires_error():
    """A failed PipelineResult must carry a non-empty error string."""
    r = PipelineResult(pipeline_id="a", pipeline_name="b", success=False)
    with pytest.raises(VerificationError, match="error required"):
        r.verify()
