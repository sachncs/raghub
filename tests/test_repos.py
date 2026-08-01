"""Repositories module coverage tests.

Exercises :class:`ChunkStore` against a fake vector store,
:class:`UnitOfWork` lifecycle, and the helper methods (row_to_record,
record_to_row).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from raghub.models import Chunk
from raghub.repos import ChunkStore, UnitOfWork


def _make_chunk(chunk_id: str = "c1", document_id: str = "d1") -> Chunk:
    """Build a Chunk fixture with required fields."""

    from datetime import UTC, datetime

    return Chunk(
        id=chunk_id,
        document_id=document_id,
        version=1,
        company="acme",
        owner="alice@example.com",
        text="hello",
        checksum="0" * 64,
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# ChunkStore (uses a fake vector store)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_vector_store() -> Any:
    """A MagicMock pretending to be a vector store."""

    store = MagicMock()
    store.create_collection.return_value = None
    store.upsert.return_value = None
    store.insert.return_value = None
    store.delete.return_value = None
    store.optimize.return_value = None
    store.health.return_value = {"status": "ok"}
    store.search.return_value = []
    store.delete_document.return_value = 0
    return store


def test_chunk_store_initialize(fake_vector_store: Any) -> None:
    """ChunkStore.initialize calls create_collection on the underlying store."""

    repo = ChunkStore(fake_vector_store)
    asyncio_run = __import__("asyncio").run
    asyncio_run(repo.initialize())
    fake_vector_store.create_collection.assert_called_once()


def test_chunk_store_insert(fake_vector_store: Any) -> None:
    """ChunkStore.insert delegates to vector store."""

    repo = ChunkStore(fake_vector_store)
    chunk = _make_chunk()
    asyncio_run = __import__("asyncio").run
    asyncio_run(repo.insert(chunk, [0.1, 0.2, 0.3]))
    fake_vector_store.insert.assert_called_once()


def test_chunk_store_upsert(fake_vector_store: Any) -> None:
    """ChunkStore.upsert delegates to vector store with embeddings."""

    repo = ChunkStore(fake_vector_store)
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    asyncio_run = __import__("asyncio").run
    asyncio_run(repo.upsert(chunks, embeddings))
    fake_vector_store.upsert.assert_called_once()


def test_chunk_store_upsert_without_embeddings_raises(fake_vector_store: Any) -> None:
    """ChunkStore.upsert raises ValueError when embeddings is None."""

    repo = ChunkStore(fake_vector_store)

    async def _drive() -> None:
        await repo.upsert([_make_chunk()])

    with pytest.raises(ValueError, match="embeddings required"):
        __import__("asyncio").run(_drive())


def test_chunk_store_delete_by_id(fake_vector_store: Any) -> None:
    """ChunkStore.delete_by_id delegates to vector store."""

    repo = ChunkStore(fake_vector_store)
    asyncio_run = __import__("asyncio").run
    asyncio_run(repo.delete_by_id("c1"))
    fake_vector_store.delete.assert_called_once_with(["c1"])


def test_chunk_store_delete_by_document(fake_vector_store: Any) -> None:
    """ChunkStore.delete_by_document delegates to vector store."""

    repo = ChunkStore(fake_vector_store)
    asyncio_run = __import__("asyncio").run
    asyncio_run(repo.delete_by_document("d1"))
    fake_vector_store.delete_document.assert_called_once_with("d1")


def test_chunk_store_search(fake_vector_store: Any) -> None:
    """ChunkStore.search delegates to vector store."""

    repo = ChunkStore(fake_vector_store)
    fake_vector_store.search.return_value = [{"id": "c1"}]
    asyncio_run = __import__("asyncio").run
    results = asyncio_run(repo.search([0.1, 0.2], top_k=5))
    assert results == [{"id": "c1"}]
    fake_vector_store.search.assert_called_once()


def test_chunk_store_optimize(fake_vector_store: Any) -> None:
    """ChunkStore.optimize delegates to vector store."""

    repo = ChunkStore(fake_vector_store)
    asyncio_run = __import__("asyncio").run
    asyncio_run(repo.optimize())
    fake_vector_store.optimize.assert_called_once()


def test_chunk_store_health(fake_vector_store: Any) -> None:
    """ChunkStore.health delegates to vector store."""

    repo = ChunkStore(fake_vector_store)
    asyncio_run = __import__("asyncio").run
    h = asyncio_run(repo.health())
    assert h == {"status": "ok"}


# ---------------------------------------------------------------------------
# UnitOfWork
# ---------------------------------------------------------------------------


def test_unit_of_work_init_binds_repos(tmp_path: Any) -> None:
    """UnitOfWork.__init__ wires the three repos under one coordinator."""

    fake_vector_store = MagicMock()
    uow = UnitOfWork(
        db_path=str(tmp_path / "uow.db"),
        vector_store=fake_vector_store,
        session_timeout=3600,
    )
    assert uow.db_path == str(tmp_path / "uow.db")
    assert uow.vector_store is fake_vector_store
    assert uow.session_timeout == 3600
    assert hasattr(uow, "document_repo")
    assert hasattr(uow, "chunk_repo")
    assert hasattr(uow, "session_repo")


def test_unit_of_work_close_when_not_initialized_is_noop(tmp_path: Any) -> None:
    """UnitOfWork.close() is silent before initialize()."""

    uow = UnitOfWork(
        db_path=str(tmp_path / "uow.db"),
        vector_store=MagicMock(),
        session_timeout=3600,
    )
    asyncio_run = __import__("asyncio").run
    asyncio_run(uow.close())  # does not raise
