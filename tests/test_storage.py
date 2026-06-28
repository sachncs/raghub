"""Storage-layer tests for document, chunk, and session persistence.

Covers both the SQLite repositories and the image store.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from raghub.models import DocumentVersion, ConversationTurn, DocumentLifecycleStatus
from raghub.storage.image_store import FilesystemImageStore


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


class TestSqliteDocumentRepository:
    @pytest.mark.asyncio
    async def test_save_and_get(self, tmp_db):
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        doc = DocumentVersion(
            document_id="doc1",
            checksum="abc123",
            owner="alice",
            organization="acme",
        )
        await repo.save(doc)
        loaded = await repo.get("doc1")
        assert loaded is not None
        assert loaded.document_id == "doc1"
        assert loaded.checksum == "abc123"
        assert loaded.owner == "alice"

    @pytest.mark.asyncio
    async def test_get_by_checksum(self, tmp_db):
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        doc = DocumentVersion(checksum="xyz789", owner="bob", organization="beta")
        await repo.save(doc)
        loaded = await repo.get_by_checksum("xyz789")
        assert loaded is not None
        assert loaded.owner == "bob"

    @pytest.mark.asyncio
    async def test_delete(self, tmp_db):
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        doc = DocumentVersion(checksum="del123", owner="alice", organization="acme")
        await repo.save(doc)
        await repo.delete(doc.document_id)
        loaded = await repo.get(doc.document_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_update_status(self, tmp_db):
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        doc = DocumentVersion(checksum="upd123", owner="alice", organization="acme")
        await repo.save(doc)
        await repo.update_status(doc.document_id, DocumentLifecycleStatus.READY)
        loaded = await repo.get(doc.document_id)
        assert loaded is not None
        assert loaded.status == DocumentLifecycleStatus.READY

    @pytest.mark.asyncio
    async def test_list_by_organization(self, tmp_db):
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        for i in range(3):
            doc = DocumentVersion(checksum=f"c{i}", owner=f"user{i}", organization="org1")
            await repo.save(doc)
        docs = await repo.list_by_organization("org1")
        assert len(docs) == 3


class TestSqliteSessionStore:
    @pytest.mark.asyncio
    async def test_create_and_get_session(self, tmp_db):
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        store = SqliteSessionStore(tmp_db, timeout_seconds=3600)
        await store.initialize()
        session = await store.create_session("user1")
        assert session.user_id == "user1"
        loaded = await store.get_session(session.session_id)
        assert loaded is not None
        assert loaded.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_get_by_token(self, tmp_db):
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        store = SqliteSessionStore(tmp_db, timeout_seconds=3600)
        await store.initialize()
        session = await store.create_session("user1")
        loaded = await store.get_by_token(session.token)
        assert loaded is not None
        assert loaded.user_id == "user1"

    @pytest.mark.asyncio
    async def test_append_history(self, tmp_db):
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        store = SqliteSessionStore(tmp_db, timeout_seconds=3600)
        await store.initialize()
        session = await store.create_session("user1")
        turn = ConversationTurn(question="Hello", answer="Hi there")
        await store.append_history(session.session_id, turn)
        history = await store.get_history(session.session_id)
        assert len(history) == 1
        assert history[0].question == "Hello"

    @pytest.mark.asyncio
    async def test_delete_session(self, tmp_db):
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        store = SqliteSessionStore(tmp_db, timeout_seconds=3600)
        await store.initialize()
        session = await store.create_session("user1")
        await store.delete_session(session.session_id)
        loaded = await store.get_session(session.session_id)
        assert loaded is None


class TestSqliteUserStore:
    @pytest.mark.asyncio
    async def test_create_and_get_user(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        store = SqliteUserStore(tmp_db)
        await store.initialize()
        user = await store.create_user("alice@test.com", "secret123")
        assert user.email == "alice@test.com"
        loaded = await store.get_by_email("alice@test.com")
        assert loaded is not None
        assert loaded.email == "alice@test.com"

    @pytest.mark.asyncio
    async def test_verify_password(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        store = SqliteUserStore(tmp_db)
        await store.initialize()
        await store.create_user("bob@test.com", "mypassword")
        result = await store.verify_password("bob@test.com", "mypassword")
        assert result is not None
        assert result.email == "bob@test.com"
        wrong = await store.verify_password("bob@test.com", "wrong")
        assert wrong is None

    @pytest.mark.asyncio
    async def test_list_users(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        store = SqliteUserStore(tmp_db)
        await store.initialize()
        await store.create_user("a@test.com", "p1")
        await store.create_user("b@test.com", "p2")
        users = await store.list_users()
        assert len(users) == 2


class TestFilesystemImageStore:
    def test_save_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemImageStore(tmpdir)
            content_hash = store.save(b"fake-image-bytes", ".png")
            assert len(content_hash) == 64
            path = store.get_path(content_hash, ".png")
            assert path is not None
            assert path.exists()
            data = store.get_bytes(content_hash, ".png")
            assert data == b"fake-image-bytes"

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemImageStore(tmpdir)
            content_hash = store.save(b"delete-me", ".png")
            assert store.delete(content_hash, ".png") is True
            assert store.delete(content_hash, ".png") is False
