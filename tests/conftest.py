"""Pytest fixtures and shared test configuration.

This module sets up a stable test environment by:

1. Defaulting :envvar:`JWT_SECRET` to a non-production value so the
   auth layer can mint tokens without an explicit secret.
2. Disabling passwordless login by default.
3. Pinning :envvar:`CORS_ORIGINS` to a non-wildcard value so the
   production guard refuses wildcard+credentials at startup; tests
   that explicitly need wildcard CORS must clear the env var.
4. Adding the repository root to :data:`sys.path` so absolute imports
   like ``from raghub.…`` resolve from the working tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from raghub.models import Classification, ChunkRecord

os.environ.setdefault("JWT_SECRET", "test-secret-must-be-32-bytes-or-longer-for-sha256")
os.environ.setdefault("RAG_ALLOW_PASSWORDLESS", "0")
os.environ.setdefault("CORS_ORIGINS", "http://testserver")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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