"""Tests for SQLite repository modules: chunk, session, and unit-of-work."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from raghub.domain import ChunkRepository, SessionRepository, UnitOfWork as BaseUnitOfWork
from raghub.models import ChunkRecord, SessionRecord
from raghub.repositories.sqlite_chunk_repo import SqliteChunkRepository
from raghub.repositories.sqlite_session_repo import SqliteSessionRepository
from raghub.repositories.unit_of_work import UnitOfWork
from raghub.storage.database import DatabaseManager
from raghub.vectorstore.base import BaseVectorStore

pytestmark = pytest.mark.asyncio


# ===========================================================================
# Helpers
# ===========================================================================


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock(spec=BaseVectorStore)
    store.create_collection = MagicMock()
    store.insert = MagicMock()
    store.upsert = MagicMock()
    store.delete = MagicMock()
    store.delete_document = MagicMock()
    store.search = MagicMock(return_value=[{"id": "c1", "score": 0.95}])
    store.optimize = MagicMock()
    store.health = MagicMock(return_value={"status": "ok"})
    return store


@pytest.fixture
def mock_db_manager() -> MagicMock:
    mgr = MagicMock(spec=DatabaseManager)
    mgr.connection = AsyncMock(spec=aiosqlite.Connection)
    mgr.connection.execute = AsyncMock()
    mgr.connection.commit = AsyncMock()
    mgr.connection.close = AsyncMock()
    mgr.connect = AsyncMock(return_value=mgr.connection)
    return mgr


@pytest.fixture
def chunk_record() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        version=1,
        company="acme",
        owner="alice",
        text="some text",
    )


# ===========================================================================
# SqliteChunkRepository
# ===========================================================================


class TestSqliteChunkRepository:
    async def test_initialize_delegates_to_store(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        await repo.initialize()
        mock_vector_store.create_collection.assert_called_once_with()

    async def test_insert_delegates_to_store(self, mock_vector_store, chunk_record) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        embedding = [0.1, 0.2, 0.3]
        await repo.insert(chunk_record, embedding)
        mock_vector_store.insert.assert_called_once_with([chunk_record], [embedding])

    async def test_upsert_with_embeddings(self, mock_vector_store, chunk_record) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        records = [chunk_record, chunk_record]
        await repo.upsert(records, embeddings)
        mock_vector_store.upsert.assert_called_once_with(records, embeddings)

    async def test_upsert_raises_when_embeddings_none(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        with pytest.raises(ValueError, match="embeddings required for upsert"):
            await repo.upsert([], None)

    async def test_delete_by_id_delegates_to_store(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        await repo.delete_by_id("chunk-1")
        mock_vector_store.delete.assert_called_once_with(["chunk-1"])

    async def test_delete_by_document_delegates_to_store(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        await repo.delete_by_document("doc-1")
        mock_vector_store.delete_document.assert_called_once_with("doc-1")

    async def test_search_delegates_to_store(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        vector = [0.1, 0.2]
        result = await repo.search(vector, top_k=5, metadata_filter="company=acme")
        mock_vector_store.search.assert_called_once_with(
            vector=vector, top_k=5, metadata_filter="company=acme"
        )
        assert result == [{"id": "c1", "score": 0.95}]

    async def test_search_no_filter(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        vector = [0.1, 0.2]
        await repo.search(vector, top_k=10)
        mock_vector_store.search.assert_called_once_with(
            vector=vector, top_k=10, metadata_filter=""
        )

    async def test_optimize_delegates_to_store(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        await repo.optimize()
        mock_vector_store.optimize.assert_called_once_with()

    async def test_health_delegates_to_store(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        result = await repo.health()
        mock_vector_store.health.assert_called_once_with()
        assert result == {"status": "ok"}

    async def test_is_chunk_repository(self, mock_vector_store) -> None:
        repo = SqliteChunkRepository(mock_vector_store)
        assert isinstance(repo, ChunkRepository)


# ===========================================================================
# SqliteSessionRepository
# ===========================================================================


class TestSqliteSessionRepository:
    @pytest.fixture
    def tmp_db(self, tmp_path: Path) -> str:
        return str(tmp_path / "test_sessions.db")

    async def test_initialize_with_db_manager(self, tmp_db: str, mock_db_manager: MagicMock) -> None:
        mock_db_manager.connection.executescript = AsyncMock()
        repo = SqliteSessionRepository(tmp_db, db_manager=mock_db_manager)
        await repo.initialize()
        mock_db_manager.connection.executescript.assert_awaited_once()

    async def test_initialize_without_db_manager(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        await repo.initialize()
        assert Path(tmp_db).exists()

    async def test_create_with_db_manager(self, tmp_db: str, mock_db_manager: MagicMock) -> None:
        mock_conn = mock_db_manager.connection
        mock_conn.execute = AsyncMock()
        repo = SqliteSessionRepository(tmp_db, db_manager=mock_db_manager)
        record = SessionRecord(
            session_id="sid-1",
            user_id="user1",
            token="tok-1",
            created_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            expires_at=datetime(2026, 6, 1, 13, 0, 0, tzinfo=timezone.utc),
            last_seen_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        await repo.create(record)
        mock_conn.execute.assert_awaited_once()
        call_args = mock_conn.execute.await_args
        assert call_args is not None
        sql = call_args[0][0] if isinstance(call_args[0], tuple) else call_args[0]
        assert "INSERT INTO sessions" in sql
        mock_conn.commit.assert_not_awaited()
        mock_conn.close.assert_not_awaited()

    async def test_create_without_db_manager(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        await repo.initialize()
        record = SessionRecord(
            session_id="sid-2",
            user_id="user2",
            token="tok-2",
            created_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            expires_at=datetime(2026, 6, 1, 13, 0, 0, tzinfo=timezone.utc),
            last_seen_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        await repo.create(record)
        loaded = await repo.get("sid-2")
        assert loaded is not None
        assert loaded.user_id == "user2"

    async def test_save_delegates_to_inner(self, tmp_db: str, mock_db_manager: MagicMock) -> None:
        repo = SqliteSessionRepository(tmp_db, db_manager=mock_db_manager)
        with patch.object(repo.inner, "update_session", new=AsyncMock()) as mock_update:
            record = SessionRecord(
                session_id="sid-3",
                user_id="user3",
                token="tok-3",
                created_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
                expires_at=datetime(2026, 6, 1, 13, 0, 0, tzinfo=timezone.utc),
                last_seen_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
            await repo.save(record)
            mock_update.assert_awaited_once_with(record)

    async def test_get_delegates_to_inner(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        await repo.initialize()
        result = await repo.get("nonexistent")
        assert result is None

    async def test_get_returns_record(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        await repo.initialize()
        created = await repo.inner.create_session("user1")
        loaded = await repo.get(created.session_id)
        assert loaded is not None
        assert loaded.user_id == "user1"

    # Simpler: test delegation via mocks
    async def test_get_delegates_to_inner_store(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        with patch.object(repo.inner, "get_session", new=AsyncMock(return_value=None)) as mock_get:
            result = await repo.get("sid-x")
            mock_get.assert_awaited_once_with("sid-x")
            assert result is None

    async def test_get_by_token_delegates_to_inner(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        with patch.object(repo.inner, "get_by_token", new=AsyncMock(return_value=None)) as mock_get:
            result = await repo.get_by_token("tok-x")
            mock_get.assert_awaited_once_with("tok-x")
            assert result is None

    async def test_delete_delegates_to_inner(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        with patch.object(repo.inner, "delete_session", new=AsyncMock()) as mock_del:
            await repo.delete("sid-x")
            mock_del.assert_awaited_once_with("sid-x")

    async def test_conn_with_db_manager(self, tmp_db: str, mock_db_manager: MagicMock) -> None:
        repo = SqliteSessionRepository(tmp_db, db_manager=mock_db_manager)
        conn = await repo.conn()
        assert conn is mock_db_manager.connection

    async def test_conn_without_db_manager_opens_new(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        conn = await repo.conn()
        assert isinstance(conn, aiosqlite.Connection)
        await conn.close()

    async def test_is_session_repository(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        assert isinstance(repo, SessionRepository)

    async def test_real_create_and_get_roundtrip(self, tmp_db: str) -> None:
        repo = SqliteSessionRepository(tmp_db, timeout_seconds=60)
        await repo.initialize()
        record = SessionRecord(
            session_id="rt-1",
            user_id="rt-user",
            token="rt-tok",
            created_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            expires_at=datetime(2026, 6, 1, 14, 0, 0, tzinfo=timezone.utc),
            last_seen_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        await repo.create(record)
        loaded = await repo.get("rt-1")
        assert loaded is not None
        assert loaded.session_id == "rt-1"
        assert loaded.user_id == "rt-user"
        assert loaded.token == "rt-tok"


# ===========================================================================
# UnitOfWork
# ===========================================================================


class TestUnitOfWork:
    async def test_initialization_sets_up_repos(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        assert uow.db_path == db_path
        assert uow.vector_store is mock_vector_store
        assert uow.session_timeout == 3600
        assert uow.initialized is False
        assert uow.document_repo is not None
        assert uow.chunk_repo is not None
        assert uow.session_repo is not None
        assert uow.db_manager is not None
        assert isinstance(uow, BaseUnitOfWork)

    async def test_initialize_connects_and_inits_all_repos(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow2.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        assert uow.initialized is True
        assert uow.db_manager.conn is not None

    async def test_initialize_is_idempotent(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow3.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        await uow.initialize()
        assert uow.initialized is True

    async def test_initialize_inits_document_repo(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow4.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        with patch.object(uow.document_repo, "initialize", new=AsyncMock()) as mock_doc_init:
            await uow.initialize()
            mock_doc_init.assert_awaited_once()

    async def test_initialize_inits_chunk_repo(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow5.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        with patch.object(uow.chunk_repo, "initialize", new=AsyncMock()) as mock_chunk_init:
            await uow.initialize()
            mock_chunk_init.assert_awaited_once()

    async def test_initialize_inits_session_repo(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow6.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        with patch.object(uow.session_repo, "initialize", new=AsyncMock()) as mock_sess_init:
            await uow.initialize()
            mock_sess_init.assert_awaited_once()

    async def test_commit_when_in_transaction(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow7.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        uow.in_transaction = True
        with patch.object(uow.db_manager.connection, "commit", new=AsyncMock()) as mock_commit:
            await uow.commit()
            mock_commit.assert_awaited_once()
            assert uow.in_transaction is False

    async def test_commit_when_not_in_transaction(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow8.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        uow.in_transaction = False
        with patch.object(uow.db_manager.connection, "commit", new=AsyncMock()) as mock_commit:
            await uow.commit()
            mock_commit.assert_not_awaited()

    async def test_rollback_when_in_transaction(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow9.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        uow.in_transaction = True
        with patch.object(uow.db_manager.connection, "rollback", new=AsyncMock()) as mock_rollback:
            await uow.rollback()
            mock_rollback.assert_awaited_once()
            assert uow.in_transaction is False

    async def test_rollback_when_not_in_transaction(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow10.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        uow.in_transaction = False
        with patch.object(uow.db_manager.connection, "rollback", new=AsyncMock()) as mock_rollback:
            await uow.rollback()
            mock_rollback.assert_not_awaited()

    async def test_async_context_manager_begins_and_commits(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow11.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        async with uow:
            assert uow.in_transaction is True
        assert uow.in_transaction is False

    async def test_async_context_manager_rollback_on_error(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow12.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.initialize()
        with patch.object(uow, "rollback", new=AsyncMock()) as mock_rollback, \
             patch.object(uow, "commit", new=AsyncMock()) as mock_commit:
            try:
                async with uow:
                    raise ValueError("test error")
            except ValueError:
                pass
            mock_rollback.assert_awaited_once()
            mock_commit.assert_not_awaited()

    async def test_db_manager_is_passed_to_repos(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow13.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        assert uow.document_repo.db_manager is uow.db_manager
        assert uow.session_repo.db_manager is uow.db_manager
        # chunk repo doesn't take a db_manager

    async def test_db_manager_connect_called_on_initialize(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow14.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        mock_conn = AsyncMock(spec=aiosqlite.Connection)
        mock_conn.executescript = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchone = AsyncMock(return_value=None)
        mock_conn.fetchall = AsyncMock(return_value=[])
        type(uow.db_manager).connection = property(lambda self: mock_conn)
        with patch.object(uow.db_manager, "connect", new=AsyncMock()) as mock_connect:
            await uow.initialize()
            mock_connect.assert_awaited_once()

    async def test_health_via_chunk_repo(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow15.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        result = await uow.chunk_repo.health()
        assert result == {"status": "ok"}

    async def test_vector_store_creates_collection(self, tmp_path: Path, mock_vector_store: MagicMock) -> None:
        db_path = str(tmp_path / "test_uow16.db")
        uow = UnitOfWork(db_path, mock_vector_store, session_timeout=3600)
        await uow.chunk_repo.initialize()
        mock_vector_store.create_collection.assert_called_once_with()
