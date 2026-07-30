"""Shared test fixtures."""

from __future__ import annotations

import pytest

from raghub.models import ChunkRecord, Classification


@pytest.fixture
def sample_chunk() -> ChunkRecord:
    """Build one minimal :class:`ChunkRecord` for tests."""
    return ChunkRecord(
        chunk_id="test-chunk-1",
        document_id="doc-1",
        version=1,
        text="Revenue grew 12 percent in Q3 2024.",
        classification=Classification.INTERNAL,
        company="acme",
        owner="alice@example.com",
        department="finance",
        checksum="abc123",
        page=0,
        source_location="page 1",
    )


@pytest.fixture
def sample_chunks(sample_chunk: ChunkRecord) -> list[ChunkRecord]:
    """Build a small batch of chunks for indexing tests."""
    return [sample_chunk]


@pytest.fixture
def sample_vectors() -> list[list[float]]:
    """Build vectors matching the default embedding_dim (384)."""
    return [[0.01 * (i + 1)] * 384 for i in range(1)]