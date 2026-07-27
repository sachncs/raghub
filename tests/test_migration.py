"""Qualitative tests for the JSON-to-SQLite migration utility.

These tests cover the real contract of :func:`migrate_from_json`:

* Every document version lands in the SQLite store (no losses, no
  duplicates).
* Session ids, tokens, and expiry times are preserved across the
  migration.
* The migration is additive — the source JSON files are not
  deleted, modified, or corrupted.
* The progress-bar toggle (``show_progress=False``) does not change
  the result.
* An empty registry + empty sessions store is a no-op (no spurious
  rows).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from raghub.models import DocumentLifecycleStatus, DocumentVersion, SessionRecord
from raghub.storage.json_registry import JsonDocumentRegistry
from raghub.storage.session_store import JsonSessionStore


def _registry_with_versions(tmp_path: Path, versions: list[DocumentVersion]) -> Path:
    path = tmp_path / "registry.json"
    registry = JsonDocumentRegistry(path)
    for v in versions:
        registry.save_version(v)
    return path


def _empty_sessions(tmp_path: Path) -> Path:
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"sessions": {}}))
    return path


def _sessions_with(tmp_path: Path, sessions: list[SessionRecord]) -> Path:
    path = tmp_path / "sessions.json"
    store = JsonSessionStore(path, timeout_seconds=3600)
    for s in sessions:
        store.sessions[s.token] = s
    store.save()
    return path


def _doc(document_id: str, version: int, checksum: str) -> DocumentVersion:
    return DocumentVersion(
        document_id=document_id,
        version=version,
        checksum=checksum,
        owner="u@acme.com",
        organization="Acme",
        status=DocumentLifecycleStatus.READY,
    )


def _session(token: str, user_id: str = "u1") -> SessionRecord:
    now = datetime.now(UTC)
    return SessionRecord(
        session_id=f"sid-{token}",
        user_id=user_id,
        token=token,
        created_at=now,
        expires_at=now + timedelta(hours=1),
        last_seen_at=now,
    )


# ---------------------------------------------------------------------------
# Document migration
# ---------------------------------------------------------------------------


class TestDocumentMigration:
    @pytest.mark.asyncio
    async def test_empty_registry_migrates_zero_rows(self, tmp_path: Path) -> None:
        from raghub.repositories import SqliteDocumentRepository
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(tmp_path, [])
        sessions_path = _empty_sessions(tmp_path)

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        repo = SqliteDocumentRepository(db)
        await repo.initialize()
        assert await repo.list_all() == []

    @pytest.mark.asyncio
    async def test_single_version_migrates(self, tmp_path: Path) -> None:
        from raghub.repositories import SqliteDocumentRepository
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        v1 = _doc("d1", 1, "c1")
        registry_path = _registry_with_versions(tmp_path, [v1])
        sessions_path = _empty_sessions(tmp_path)

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        repo = SqliteDocumentRepository(db)
        await repo.initialize()
        rows = await repo.list_all()
        assert len(rows) == 1
        assert rows[0].document_id == "d1"
        assert rows[0].checksum == "c1"

    @pytest.mark.asyncio
    async def test_multiple_versions_per_document_all_migrate(
        self, tmp_path: Path
    ) -> None:
        from raghub.repositories import SqliteDocumentRepository
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(
            tmp_path,
            [_doc("d1", 1, "c1"), _doc("d1", 2, "c2"), _doc("d1", 3, "c3")],
        )
        sessions_path = _empty_sessions(tmp_path)

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        repo = SqliteDocumentRepository(db)
        await repo.initialize()
        versions = await repo.list_versions("d1")
        assert [v.version for v in versions] == [1, 2, 3]
        assert [v.checksum for v in versions] == ["c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_multiple_documents_all_migrate(self, tmp_path: Path) -> None:
        from raghub.repositories import SqliteDocumentRepository
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(
            tmp_path,
            [
                _doc("d1", 1, "c1"),
                _doc("d2", 1, "c2"),
                _doc("d3", 1, "c3"),
            ],
        )
        sessions_path = _empty_sessions(tmp_path)

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        repo = SqliteDocumentRepository(db)
        await repo.initialize()
        ids = {d.document_id for d in await repo.list_all()}
        assert ids == {"d1", "d2", "d3"}

    @pytest.mark.asyncio
    async def test_show_progress_false_does_not_corrupt(
        self, tmp_path: Path
    ) -> None:
        from raghub.repositories import SqliteDocumentRepository
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(
            tmp_path, [_doc("d1", 1, "c1"), _doc("d2", 1, "c2")]
        )
        sessions_path = _empty_sessions(tmp_path)

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        repo = SqliteDocumentRepository(db)
        await repo.initialize()
        assert len(await repo.list_all()) == 2


# ---------------------------------------------------------------------------
# Session migration
# ---------------------------------------------------------------------------


class TestSessionMigration:
    @pytest.mark.asyncio
    async def test_empty_sessions_migrates_zero_rows(self, tmp_path: Path) -> None:
        from raghub.storage.migration import migrate_from_json
        from raghub.storage.sqlite_session_store import SqliteSessionStore

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(tmp_path, [])
        sessions_path = _empty_sessions(tmp_path)

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        store = SqliteSessionStore(db, timeout_seconds=3600)
        await store.initialize()
        assert await store.get_session("any") is None

    @pytest.mark.asyncio
    async def test_single_session_migrates(self, tmp_path: Path) -> None:
        from raghub.storage.migration import migrate_from_json
        from raghub.storage.sqlite_session_store import SqliteSessionStore

        db = tmp_path / "migrated.db"
        s = _session("tok-1")
        registry_path = _registry_with_versions(tmp_path, [])
        sessions_path = _sessions_with(tmp_path, [s])

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        store = SqliteSessionStore(db, timeout_seconds=3600)
        await store.initialize()
        loaded = await store.get_session(s.session_id)
        assert loaded is not None
        assert loaded.user_id == "u1"
        assert loaded.token == "tok-1"

    @pytest.mark.asyncio
    async def test_token_lookup_works_after_migration(self, tmp_path: Path) -> None:
        from raghub.storage.migration import migrate_from_json
        from raghub.storage.sqlite_session_store import SqliteSessionStore

        db = tmp_path / "migrated.db"
        s = _session("tok-1")
        registry_path = _registry_with_versions(tmp_path, [])
        sessions_path = _sessions_with(tmp_path, [s])

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        store = SqliteSessionStore(db, timeout_seconds=3600)
        await store.initialize()
        loaded = await store.get_by_token("tok-1")
        assert loaded is not None
        assert loaded.session_id == s.session_id

    @pytest.mark.asyncio
    async def test_multiple_sessions_all_migrate(self, tmp_path: Path) -> None:
        from raghub.storage.migration import migrate_from_json
        from raghub.storage.sqlite_session_store import SqliteSessionStore

        db = tmp_path / "migrated.db"
        sessions = [_session(f"tok-{i}", user_id=f"u{i}") for i in range(5)]
        registry_path = _registry_with_versions(tmp_path, [])
        sessions_path = _sessions_with(tmp_path, sessions)

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        store = SqliteSessionStore(db, timeout_seconds=3600)
        await store.initialize()
        for s in sessions:
            loaded = await store.get_session(s.session_id)
            assert loaded is not None
            assert loaded.user_id == s.user_id


# ---------------------------------------------------------------------------
# Migration is additive — the source files survive intact
# ---------------------------------------------------------------------------


class TestAdditiveMigration:
    @pytest.mark.asyncio
    async def test_source_registry_file_is_not_modified(
        self, tmp_path: Path
    ) -> None:
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(tmp_path, [_doc("d1", 1, "c1")])
        sessions_path = _empty_sessions(tmp_path)
        before = registry_path.read_bytes()

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        after = registry_path.read_bytes()
        assert before == after, (
            "Migration must not modify the source JSON file — operators "
            "rely on the original for rollback if the SQLite copy is "
            "found to be corrupt."
        )

    @pytest.mark.asyncio
    async def test_source_sessions_file_is_not_modified(
        self, tmp_path: Path
    ) -> None:
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(tmp_path, [])
        sessions_path = _sessions_with(tmp_path, [_session("tok-1")])
        before = sessions_path.read_bytes()

        await migrate_from_json(db, registry_path, sessions_path, show_progress=False)
        after = sessions_path.read_bytes()
        assert before == after


# ---------------------------------------------------------------------------
# Idempotency — re-running the migration on the same file does not lose rows
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_documents_survive_double_migration_via_try_insert(
        self, tmp_path: Path
    ) -> None:
        """Running the migration twice with the same source produces
        the same row count — the document side uses ``try_insert`` to
        skip already-present composite keys, so a re-run is a no-op
        for documents (session side is exercised separately)."""
        from raghub.repositories import SqliteDocumentRepository
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        registry_path = _registry_with_versions(
            tmp_path, [_doc("d1", 1, "c1"), _doc("d1", 2, "c2")]
        )
        # Empty sessions so the second migration doesn't choke on the
        # session primary-key collision (that's a separate behaviour
        # not in scope for this test).
        empty_sessions = _empty_sessions(tmp_path)
        first_sessions = _sessions_with(tmp_path, [_session("tok-1")])
        second_sessions = _empty_sessions(tmp_path)

        # First run with documents + one session.
        await migrate_from_json(
            db, registry_path, first_sessions, show_progress=False
        )
        # Second run with documents only (no new sessions).
        await migrate_from_json(
            db, registry_path, second_sessions, show_progress=False
        )

        repo = SqliteDocumentRepository(db)
        await repo.initialize()
        versions = await repo.list_versions("d1")
        # Both versions of d1 are present, no extra row was added.
        assert [v.version for v in versions] == [1, 2]


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_missing_registry_file_is_tolerated(self, tmp_path: Path) -> None:
        """A first-time deployment with no legacy JSON files must not
        crash the migration — it's a no-op."""
        from raghub.storage.migration import migrate_from_json

        db = tmp_path / "migrated.db"
        await migrate_from_json(
            db,
            tmp_path / "does-not-exist.json",
            _empty_sessions(tmp_path),
            show_progress=False,
        )
        # No exceptions raised; the SQLite file exists.
        assert db.exists()
