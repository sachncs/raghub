"""Shared test fixtures."""

from __future__ import annotations

import pytest

from raghub.models import Chunk, Classification


@pytest.fixture
def sample_chunk() -> Chunk:
    """Build one minimal :class:`Chunk` for tests."""
    return Chunk(
        id="test-chunk-1",
        document_id="doc-1",
        version=1,
        text="Revenue grew 12 percent in Q3 2024.",
        classification=Classification.Internal,
        company="acme",
        owner="alice@example.com",
        department="finance",
        checksum="9e3530a3ac3b60c19ce7a2f9d8c0314e405782a1ca63566000004f3cd3abbf1c",
        page=0,
        source_location="page 1",
    )


@pytest.fixture
def sample_chunks(sample_chunk: Chunk) -> list[Chunk]:
    """Build a small batch of chunks for indexing tests."""
    return [sample_chunk]


@pytest.fixture
def sample_vectors() -> list[list[float]]:
    """Build vectors matching the default embedding_dim (384)."""
    return [[0.01 * (i + 1)] * 384 for i in range(1)]
