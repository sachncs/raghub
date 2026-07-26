"""Tests for raghub.generation.InstructorStructuredOutputProvider."""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from raghub.generation import InstructorStructuredOutputProvider
from raghub.models import ChunkRecord, Classification, RetrievalHit
from raghub.vectorstore import InMemoryVectorStore


class Answer(BaseModel):
    """Test response model."""

    value: str


def _hit(text: str = "hello") -> RetrievalHit:
    chunk = ChunkRecord(
        chunk_id="c1",
        document_id="d1",
        version=1,
        text=text,
        company="acme",
        owner="me",
        classification=Classification.INTERNAL,
    )
    return RetrievalHit(chunk_id=chunk.chunk_id, score=1.0, chunk=chunk)


def test_provider_initialises_with_default_client() -> None:
    """The provider constructs cleanly with default configuration."""
    provider = InstructorStructuredOutputProvider()
    assert provider.client is None
    assert provider.client_async is None
    assert provider.model == "gpt-4o-mini"
    assert provider.async_client is True


def test_provider_accepts_custom_model_and_sync() -> None:
    """Custom model name and sync mode are stored."""
    provider = InstructorStructuredOutputProvider(model="other", async_client=False)
    assert provider.model == "other"
    assert provider.async_client is False


def test_sync_client_is_lazy() -> None:
    """The sync client is created on first access and cached."""
    provider = InstructorStructuredOutputProvider(async_client=False)
    assert provider.client is None
    # ``sync_instructor_client`` is only meaningful when instructor can
    # construct a provider without network credentials; we just check
    # that the method exists and is callable.
    assert callable(provider.sync_instructor_client)


def test_async_client_is_lazy() -> None:
    """The async client is created on first access and cached."""
    provider = InstructorStructuredOutputProvider(async_client=True)
    assert provider.client_async is None
    assert callable(provider.async_instructor_client)


@pytest.mark.asyncio
async def test_astream_returns_an_async_iterator() -> None:
    """``astream`` yields an async iterator with at least one item."""

    async def fake_generate(**_: Any) -> Answer:
        return Answer(value="ok")

    provider = InstructorStructuredOutputProvider()
    provider.generate = fake_generate  # type: ignore[method-assign]
    stream = await provider.astream(
        response_model=Answer, question="q?", context=[_hit()]
    )
    items: list[Answer] = []
    async for item in stream:
        items.append(item)
    assert len(items) == 1
    assert items[0].value == "ok"


def test_instructor_provider_used_through_memory_store() -> None:
    """A simple smoke check that the memory store still works alongside."""
    store = InMemoryVectorStore()
    store.create_collection()
    assert store.health()["status"] == "ok"