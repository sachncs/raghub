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
5. Patching :func:`raghub.rag.facade.default_llm` so that every test that
   constructs a :class:`RAG` without an explicit LLM or API key gets
   a deterministic stub instead of raising ``ConfigurationError``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from raghub.models import Chunk, Classification

os.environ.setdefault("JWT_SECRET", "test-secret-must-be-32-bytes-or-longer-for-sha256")
os.environ.setdefault("RAG_ALLOW_PASSWORDLESS", "0")
os.environ.setdefault("CORS_ORIGINS", "http://testserver")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Deterministic clock
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_clock() -> Callable[[], datetime]:
    """Return a callable that always returns a fixed instant.

    Use this fixture in place of ``datetime.now(UTC)`` so that tests
    asserting timestamp equality do not flake across runs.

    Returns:
        A no-argument callable yielding ``datetime(2026, 1, 1, tzinfo=UTC)``.

    """
    instant = datetime(2026, 1, 1, tzinfo=UTC)

    def now() -> datetime:
        """Return the fixed instant."""
        return instant

    return now


@pytest.fixture
def monotonic_clock() -> Iterator[Callable[[], datetime]]:
    """Return a callable that advances by 1 second on each call.

    Useful for tests that simulate time-based transitions (e.g. session
    expiry sweeps) without sleeping.

    Yields:
        A no-argument callable that returns successive 1-second-apart
        :class:`datetime` values starting at ``datetime(2026, 1, 1, UTC)``.

    """
    counter = {"n": 0}
    base = datetime(2026, 1, 1, tzinfo=UTC)

    def now() -> datetime:
        counter["n"] += 1
        return base + timedelta(seconds=counter["n"])

    yield now


# ---------------------------------------------------------------------------
# Stub LLM for tests that construct RAG() without a real API key.
# ---------------------------------------------------------------------------


class StubLLM:
    """Deterministic LLM stub that returns a fixed answer."""

    model_name: str = "test-stub"

    def generate(self, request: object) -> str:
        return "stub answer"

    async def async_generate(self, request: object) -> str:
        return "stub answer"


def stubbed_default_llm(llm_model: str) -> StubLLM:
    """Return the stub LLM regardless of configuration."""
    return StubLLM()


@pytest.fixture(autouse=True)
def stub_default_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``raghub.rag.facade.default_llm`` so RAG() works without an API key."""
    monkeypatch.setattr("raghub.rag.facade.default_llm", stubbed_default_llm)


@pytest.fixture
def sample_chunk() -> Chunk:
    """Build one minimal :class:`Chunk` for tests."""
    import hashlib as _hashlib

    text = "Revenue grew 12 percent in Q3 2024."
    return Chunk(
        id="test-chunk-1",
        document_id="doc-1",
        version=1,
        text=text,
        classification=Classification.Internal,
        company="acme",
        owner="alice@example.com",
        department="finance",
        # Compute the checksum so Chunk.verify() (R8) passes.
        checksum=_hashlib.sha256(text.encode("utf-8")).hexdigest(),
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
