"""Storage-layer tests for document, chunk, and session persistence.

Covers both the SQLite repositories and the image store.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from raghub.models import ConversationTurn, DocumentLifecycleStatus, DocumentVersion
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


# ---------------------------------------------------------------------------
# Document version semantics, schema migration, concurrent checksum dedup
# ---------------------------------------------------------------------------


def make_doc(document_id: str, version: int, checksum: str, **overrides) -> DocumentVersion:
    defaults = dict(
        document_id=document_id,
        version=version,
        checksum=checksum,
        owner="alice",
        organization="acme",
    )
    defaults.update(overrides)
    return DocumentVersion(**defaults)


class TestDocumentVersioning:
    @pytest.mark.asyncio
    async def test_save_supports_multiple_versions(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1"))
        await repo.save(make_doc("doc1", 2, "c2"))
        versions = await repo.list_versions("doc1")
        assert [v.version for v in versions] == [1, 2]
        assert [v.checksum for v in versions] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_get_returns_latest_version(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1"))
        await repo.save(make_doc("doc1", 2, "c2"))
        loaded = await repo.get("doc1")
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.checksum == "c2"

    @pytest.mark.asyncio
    async def test_get_version_returns_specific_version(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1"))
        await repo.save(make_doc("doc1", 2, "c2"))
        v1 = await repo.get_version("doc1", 1)
        v2 = await repo.get_version("doc1", 2)
        assert v1 is not None and v2 is not None
        assert v1.checksum == "c1"
        assert v2.checksum == "c2"

    @pytest.mark.asyncio
    async def test_save_upsert_replaces_same_version(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1"))
        await repo.save(make_doc("doc1", 1, "c1-updated"))
        versions = await repo.list_versions("doc1")
        assert len(versions) == 1
        assert versions[0].checksum == "c1-updated"

    @pytest.mark.asyncio
    async def test_delete_removes_all_versions(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1"))
        await repo.save(make_doc("doc1", 2, "c2"))
        await repo.delete("doc1")
        assert await repo.list_versions("doc1") == []

    @pytest.mark.asyncio
    async def test_delete_version_removes_only_that_version(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1"))
        await repo.save(make_doc("doc1", 2, "c2"))
        await repo.delete_version("doc1", 1)
        versions = await repo.list_versions("doc1")
        assert [v.version for v in versions] == [2]

    @pytest.mark.asyncio
    async def test_update_status_only_touches_latest(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1", status=DocumentLifecycleStatus.READY))
        await repo.save(make_doc("doc1", 2, "c2", status=DocumentLifecycleStatus.READY))
        await repo.update_status("doc1", DocumentLifecycleStatus.FAILED)
        v1 = await repo.get_version("doc1", 1)
        v2 = await repo.get_version("doc1", 2)
        assert v1 is not None and v2 is not None
        assert v1.status == DocumentLifecycleStatus.READY
        assert v2.status == DocumentLifecycleStatus.FAILED

    @pytest.mark.asyncio
    async def test_list_by_organization_returns_latest_per_doc(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1", organization="org1"))
        await repo.save(make_doc("doc1", 2, "c2", organization="org1"))
        await repo.save(make_doc("doc2", 1, "c3", organization="org1"))
        listed = await repo.list_by_organization("org1")
        assert {d.document_id for d in listed} == {"doc1", "doc2"}
        doc1 = next(d for d in listed if d.document_id == "doc1")
        assert doc1.version == 2


class TestConcurrentChecksumDedup:
    @pytest.mark.asyncio
    async def test_try_insert_detects_duplicate_checksum(self, tmp_db: str) -> None:
        import aiosqlite

        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.try_insert(make_doc("doc1", 1, "dup-checksum"))
        # Different doc id, same checksum → UNIQUE(checksum) collides.
        with pytest.raises(aiosqlite.IntegrityError):
            await repo.try_insert(make_doc("doc2", 1, "dup-checksum"))

    @pytest.mark.asyncio
    async def test_try_insert_detects_duplicate_primary_key(self, tmp_db: str) -> None:
        import aiosqlite

        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.try_insert(make_doc("doc1", 1, "c1"))
        with pytest.raises(aiosqlite.IntegrityError):
            await repo.try_insert(make_doc("doc1", 1, "c1"))

    @pytest.mark.asyncio
    async def test_concurrent_try_insert_only_one_wins(self, tmp_db: str) -> None:
        import aiosqlite

        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        record_a = make_doc("docA", 1, "race-checksum")
        record_b = make_doc("docB", 1, "race-checksum")

        results = await asyncio.gather(
            repo.try_insert(record_a),
            repo.try_insert(record_b),
            return_exceptions=True,
        )
        successes = [r for r in results if r is True]
        failures = [r for r in results if isinstance(r, aiosqlite.IntegrityError)]
        assert len(successes) == 1
        assert len(failures) == 1

    @pytest.mark.asyncio
    async def test_failed_version_can_be_retried_at_higher_version(self, tmp_db: str) -> None:
        # Simulates an upload that fails mid-flight: v1 is written,
        # the upload fails and the caller retries as v2 with corrected
        # fields. Both versions must coexist.
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        v1 = make_doc("doc1", 1, "c1", status=DocumentLifecycleStatus.FAILED, error="boom")
        await repo.save(v1)
        v2 = make_doc("doc1", 2, "c2", status=DocumentLifecycleStatus.READY)
        await repo.save(v2)
        listed = await repo.list_versions("doc1")
        assert [v.version for v in listed] == [1, 2]
        assert listed[0].error == "boom"
        assert listed[1].status == DocumentLifecycleStatus.READY


class TestDocumentSchemaMigration:
    @pytest.mark.asyncio
    async def test_legacy_single_pk_db_is_migrated_with_data(self, tmp_db: str) -> None:
        # Bootstrap a legacy database: PRIMARY KEY (document_id) only,
        # no UNIQUE(checksum). After initialize() the schema must be
        # rebuilt and the existing row preserved.
        import aiosqlite

        async with aiosqlite.connect(tmp_db) as conn:
            await conn.executescript("""
                CREATE TABLE documents (
                    document_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    department TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    classification TEXT DEFAULT 'internal',
                    visibility TEXT DEFAULT 'organization',
                    status TEXT DEFAULT 'NEW',
                    filename TEXT DEFAULT '',
                    file_type TEXT DEFAULT '',
                    mime_type TEXT DEFAULT '',
                    chunk_count INTEGER DEFAULT 0,
                    chunk_ids TEXT DEFAULT '[]',
                    error TEXT
                );
                INSERT INTO documents (document_id, version, checksum, created_at, updated_at,
                                       owner, organization)
                VALUES ('legacy-1', 1, 'legacy-cksum', '2026-01-01T00:00:00+00:00',
                        '2026-01-01T00:00:00+00:00', 'alice', 'acme');
            """)
            await conn.commit()

        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()

        loaded = await repo.get("legacy-1")
        assert loaded is not None
        assert loaded.checksum == "legacy-cksum"
        assert loaded.organization == "acme"
        assert loaded.version == 1

        # New schema is in place: composite PK + UNIQUE checksum.
        async with aiosqlite.connect(tmp_db) as conn:
            cursor = await conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'"
            )
            row = await cursor.fetchone()
            assert "PRIMARY KEY (document_id, version)" in (row[0] or "")
            cursor = await conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name='ux_documents_checksum'"
            )
            idx = await cursor.fetchone()
            assert idx is not None and "UNIQUE" in (idx[0] or "")

    @pytest.mark.asyncio
    async def test_migration_normalises_zero_version(self, tmp_db: str) -> None:
        # Some legacy rows may have version=0; the migration must bump
        # them to 1 so the new NOT NULL semantics are honoured without
        # data loss.
        import aiosqlite

        async with aiosqlite.connect(tmp_db) as conn:
            await conn.executescript("""
                CREATE TABLE documents (
                    document_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    department TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    classification TEXT DEFAULT 'internal',
                    visibility TEXT DEFAULT 'organization',
                    status TEXT DEFAULT 'NEW',
                    filename TEXT DEFAULT '',
                    file_type TEXT DEFAULT '',
                    mime_type TEXT DEFAULT '',
                    chunk_count INTEGER DEFAULT 0,
                    chunk_ids TEXT DEFAULT '[]',
                    error TEXT
                );
                INSERT INTO documents (document_id, version, checksum, created_at, updated_at,
                                       owner, organization)
                VALUES ('d-zero', 0, 'c-zero', '2026-01-01T00:00:00+00:00',
                        '2026-01-01T00:00:00+00:00', 'alice', 'acme');
            """)
            await conn.commit()

        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()

        loaded = await repo.get("d-zero")
        assert loaded is not None
        assert loaded.version == 1

    @pytest.mark.asyncio
    async def test_idempotent_initialize_does_not_lose_data(self, tmp_db: str) -> None:
        from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

        repo = SqliteDocumentRepository(tmp_db)
        await repo.initialize()
        await repo.save(make_doc("doc1", 1, "c1"))
        await repo.save(make_doc("doc1", 2, "c2"))
        # Re-run initialize — must not rebuild now that schema is new.
        await repo.initialize()
        versions = await repo.list_versions("doc1")
        assert [v.version for v in versions] == [1, 2]
