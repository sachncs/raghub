"""Tests for pydantic model validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from raghub.models import (
    ChunkRecord,
    Citation,
    PipelineResult,
    Response,
    SearchResult,
)


def test_chunkrecord_requires_checksum():
    with pytest.raises(ValidationError, match="checksum"):
        ChunkRecord(
            chunk_id="x",
            document_id="d",
            version=1,
            text="hello",
            company="",
            owner="",
        )


def test_retrieval_hit_requires_matching_chunk_id(sample_chunk):
    with pytest.raises(ValidationError, match="does not match"):
        SearchResult(chunk_id="wrong-id", score=0.5, chunk=sample_chunk)


def test_response_citation_must_match_source_chunk(sample_chunk):
    sample_chunk.chunk_id = "abc"
    citation = Citation(chunk_id="xyz", document_id="d", version=1)
    with pytest.raises(ValidationError, match="not present"):
        Response(
            answer="x",
            citations=[citation],
            source_chunks=[
                SearchResult(chunk_id="abc", score=0.5, chunk=sample_chunk)
            ],
        )


def test_pipeline_result_requires_error_on_failure():
    with pytest.raises(ValidationError, match="error is required"):
        PipelineResult(pipeline_id="a", pipeline_name="b", success=False)