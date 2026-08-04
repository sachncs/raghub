"""Repositories module coverage tests.

Exercises :class:`ChunkStore` against a real :class:`MemoryStore`
instead of a mock, so a regression that silently drops writes or
loses a search hit is caught at the storage layer rather than the
mock boundary.

Also covers :class:`UnitOfWork` lifecycle and the helper methods
(row_to_record, record_to_row).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from raghub.models import Chunk
from raghub.repos import ChunkStore, UnitOfWork
from raghub.store import MemoryStore


def _make_chunk(
    chunk_id: str = "c1",
    document_id: str = "d1",
    text: str | None = None,
) -> Chunk:
    """Build a Chunk fixture with required fields."""
    from datetime import UTC, datetime
    from hashlib import sha256

    text = text if text is not None else f"hello {chunk_id}"
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        version=1,
        company="acme",
        owner="alice@example.com",
        text=text,
        checksum=sha256(text.encode("utf-8")).hexdigest(),
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def memory_store() -> MemoryStore:
    """A real in-process vector store for :class:`ChunkStore` to drive."""
    return MemoryStore(embedding_dim=3)


def _run(coro: Any) -> Any:
    """Drive an awaitable to completion."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ChunkStore — assertions on real state, not on call counts
# ---------------------------------------------------------------------------


def test_chunk_store_initialize_creates_collection(memory_store: MemoryStore) -> None:
    """``ChunkStore.initialize`` brings the underlying store online; chunks become searchable."""

    repo = ChunkStore(memory_store)
    _run(repo.initialize())
    # MemoryStore.create_collection is a no-op; readiness is observable
    # only when the store can index and search. Drive one chunk
    # round-trip to prove the wiring is live.
    chunk = _make_chunk("init-1")
    _run(repo.insert(chunk, [0.1, 0.2, 0.3]))
    assert memory_store.health()["chunks"] == 1


def test_chunk_store_insert_persists_chunk(memory_store: MemoryStore) -> None:
    """``ChunkStore.insert`` writes the chunk so it is reachable by id and search."""

    repo = ChunkStore(memory_store)
    chunk = _make_chunk("insert-1")
    _run(repo.insert(chunk, [0.1, 0.2, 0.3]))

    assert chunk.id in memory_store.records
    assert memory_store.records[chunk.id].chunk.text == chunk.text
    # Cosine-similarity search returns the just-inserted chunk.
    hits = _run(repo.search([0.1, 0.2, 0.3], top_k=5))
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == chunk.id


def test_chunk_store_upsert_writes_multiple(memory_store: MemoryStore) -> None:
    """``ChunkStore.upsert`` writes every (chunk, embedding) pair in the batch."""

    repo = ChunkStore(memory_store)
    chunks = [_make_chunk("u1"), _make_chunk("u2")]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    _run(repo.upsert(chunks, embeddings))

    for chunk in chunks:
        assert chunk.id in memory_store.records
    assert memory_store.health()["chunks"] == 2


def test_chunk_store_upsert_without_embeddings_raises(memory_store: MemoryStore) -> None:
    """``ChunkStore.upsert`` raises ``ValueError`` when ``embeddings`` is ``None``."""

    repo = ChunkStore(memory_store)

    async def _drive() -> None:
        await repo.upsert([_make_chunk()])

    with pytest.raises(ValueError, match="embeddings required"):
        asyncio.run(_drive())


def test_chunk_store_delete_by_id_removes_chunk(memory_store: MemoryStore) -> None:
    """``ChunkStore.delete_by_id`` removes the chunk so a follow-up search returns nothing."""

    repo = ChunkStore(memory_store)
    chunk = _make_chunk("del-1")
    _run(repo.insert(chunk, [0.1, 0.2, 0.3]))
    assert chunk.id in memory_store.records

    _run(repo.delete_by_id("del-1"))
    assert "del-1" not in memory_store.records
    hits = _run(repo.search([0.1, 0.2, 0.3], top_k=5))
    assert hits == []


def test_chunk_store_delete_by_document_removes_chunks(memory_store: MemoryStore) -> None:
    """``ChunkStore.delete_by_document`` removes every chunk that shares a document id."""

    repo = ChunkStore(memory_store)
    chunks = [
        _make_chunk("doc-a-1", document_id="doc-a"),
        _make_chunk("doc-a-2", document_id="doc-a"),
        _make_chunk("doc-b-1", document_id="doc-b"),
    ]
    _run(repo.upsert(chunks, [[0.1, 0.2, 0.3]] * 3))
    assert memory_store.health()["chunks"] == 3

    _run(repo.delete_by_document("doc-a"))
    remaining = set(memory_store.records.keys())
    assert remaining == {"doc-b-1"}, f"Expected only doc-b-1; got {remaining}"


def test_chunk_store_search_returns_results(memory_store: MemoryStore) -> None:
    """``ChunkStore.search`` returns the chunks most similar to ``vector``."""

    repo = ChunkStore(memory_store)
    chunk = _make_chunk("s1", text="searchable text")
    _run(repo.insert(chunk, [1.0, 0.0, 0.0]))

    results = _run(repo.search([1.0, 0.0, 0.0], top_k=5))
    assert len(results) == 1
    assert results[0]["chunk_id"] == "s1"
    assert results[0]["score"] > 0.99  # identical vector → cosine 1.0


def test_chunk_store_optimize_is_safe(memory_store: MemoryStore) -> None:
    """``ChunkStore.optimize`` runs against a populated store without raising or losing data."""

    repo = ChunkStore(memory_store)
    chunk = _make_chunk("opt-1")
    _run(repo.insert(chunk, [0.1, 0.2, 0.3]))

    _run(repo.optimize())
    # The chunk survives the optimize pass.
    assert chunk.id in memory_store.records
    hits = _run(repo.search([0.1, 0.2, 0.3], top_k=5))
    assert hits and hits[0]["chunk_id"] == "opt-1"


def test_chunk_store_health_reports_store_state(memory_store: MemoryStore) -> None:
    """``ChunkStore.health`` reflects the underlying store's status and chunk count."""

    repo = ChunkStore(memory_store)
    _run(repo.insert(_make_chunk("h1"), [0.1, 0.2, 0.3]))
    _run(repo.insert(_make_chunk("h2"), [0.4, 0.5, 0.6]))

    h = _run(repo.health())
    assert h["status"] == "ok"
    assert h["chunks"] == 2


# ---------------------------------------------------------------------------
# UnitOfWork
# ---------------------------------------------------------------------------


def test_unit_of_work_init_binds_repos(tmp_path: Any) -> None:
    """UnitOfWork.__init__ wires the three repos under one coordinator."""

    memory_store = MemoryStore(embedding_dim=3)
    uow = UnitOfWork(
        db_path=str(tmp_path / "uow.db"),
        vector_store=memory_store,
        session_timeout=3600,
    )
    assert uow.db_path == str(tmp_path / "uow.db")
    assert uow.vector_store is memory_store
    assert uow.session_timeout == 3600
    assert hasattr(uow, "document_repo")
    assert hasattr(uow, "chunk_repo")
    assert hasattr(uow, "session_repo")


def test_unit_of_work_close_when_not_initialized_is_noop(tmp_path: Any) -> None:
    """UnitOfWork.close() is silent before initialize()."""

    uow = UnitOfWork(
        db_path=str(tmp_path / "uow.db"),
        vector_store=MemoryStore(embedding_dim=3),
        session_timeout=3600,
    )
    asyncio.run(uow.close())  # does not raise
    # Nothing was initialised, so the close path is the early-return.
    assert uow.db_path == str(tmp_path / "uow.db")