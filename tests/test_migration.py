from __future__ import annotations

import asyncio
import json
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from raghub.models import DocumentLifecycleStatus, DocumentRecord


def test_migrate_from_json_empty_registries(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    registry_path = tmp_path / "registry.json"
    sessions_path = tmp_path / "sessions.json"

    registry_path.write_text(json.dumps({"documents": {}}), encoding="utf-8")
    sessions_path.write_text(json.dumps({"sessions": {}}), encoding="utf-8")

    with (
        patch("raghub.repositories.sqlite_document_repo.SqliteDocumentRepository") as mock_repo_cls,
        patch(
            "raghub.repositories.sqlite_session_repo.SqliteSessionRepository"
        ) as mock_sess_repo_cls,
        patch("raghub.storage.json_registry.JsonDocumentRegistry") as mock_json_reg_cls,
        patch("raghub.storage.session_store.JsonSessionStore") as mock_json_ss_cls,
    ):
        mock_repo = AsyncMock()
        mock_repo_cls.return_value = mock_repo
        mock_json_reg = MagicMock()
        mock_json_reg.documents = {}
        mock_json_reg_cls.return_value = mock_json_reg
        mock_sess_repo = AsyncMock()
        mock_sess_repo_cls.return_value = mock_sess_repo
        mock_json_ss = MagicMock()
        mock_json_ss.sessions = {}
        mock_json_ss_cls.return_value = mock_json_ss

        from raghub.storage.migration import migrate_from_json

        asyncio.run(migrate_from_json(str(db_path), str(registry_path), str(sessions_path)))
        mock_repo.initialize.assert_awaited_once()
        mock_sess_repo.initialize.assert_awaited_once()


def test_migrate_from_json_with_data(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    registry_path = tmp_path / "registry.json"
    sessions_path = tmp_path / "sessions.json"

    registry_path.write_text(json.dumps({"documents": {}}), encoding="utf-8")
    sessions_path.write_text(json.dumps({"sessions": {}}), encoding="utf-8")

    with (
        patch("raghub.repositories.sqlite_document_repo.SqliteDocumentRepository") as mock_repo_cls,
        patch(
            "raghub.repositories.sqlite_session_repo.SqliteSessionRepository"
        ) as mock_sess_repo_cls,
        patch("raghub.storage.json_registry.JsonDocumentRegistry") as mock_json_reg_cls,
        patch("raghub.storage.session_store.JsonSessionStore") as mock_json_ss_cls,
    ):
        mock_repo = AsyncMock()
        mock_repo_cls.return_value = mock_repo
        mock_doc = DocumentRecord(
            document_id="d1",
            version=1,
            checksum="abc",
            owner="a@b.com",
            organization="Acme",
            status=DocumentLifecycleStatus.READY,
        )
        mock_json_reg = MagicMock()
        mock_json_reg.documents = {"d1": [mock_doc]}
        mock_json_reg_cls.return_value = mock_json_reg
        mock_sess_repo = AsyncMock()
        mock_sess_repo_cls.return_value = mock_sess_repo
        mock_session = MagicMock(session_id="s1", user_id="u1", token="tok1")
        mock_json_ss = MagicMock()
        mock_json_ss.sessions = {"s1": mock_session}
        mock_json_ss_cls.return_value = mock_json_ss

        from raghub.storage.migration import migrate_from_json

        asyncio.run(migrate_from_json(str(db_path), str(registry_path), str(sessions_path)))
        mock_repo.initialize.assert_awaited_once()
        mock_sess_repo.initialize.assert_awaited_once()
        mock_repo.save.assert_awaited_once_with(mock_doc)
        # Sessions are inserted (not UPDATE'd on a missing row); each
        # JSON session must produce one INSERT call.
        mock_sess_repo.create.assert_awaited_once_with(mock_session)


def test_migrate_from_json_inserts_sessions_into_sqlite(tmp_path: Path) -> None:
    """End-to-end: real JSON files migrate to SQLite rows that survive reopen."""
    from datetime import datetime, timedelta

    from raghub.models import SessionRecord
    from raghub.storage.migration import migrate_from_json
    from raghub.storage.session_store import JsonSessionStore

    db_path = tmp_path / "migrated.db"
    registry_path = tmp_path / "registry.json"
    sessions_path = tmp_path / "sessions.json"

    registry_path.write_text(json.dumps({"documents": {}}), encoding="utf-8")

    json_sessions = JsonSessionStore(sessions_path, timeout_seconds=3600)
    now = datetime.now(UTC)
    session = SessionRecord(
        session_id="sid-real",
        user_id="u-real",
        token="tok-real",
        created_at=now,
        expires_at=now + timedelta(hours=1),
        last_seen_at=now,
    )
    json_sessions.sessions[session.token] = session
    json_sessions.save()

    asyncio.run(migrate_from_json(str(db_path), str(registry_path), str(sessions_path)))

    # Re-open the SQLite and confirm the session landed.
    from raghub.storage.sqlite_session_store import SqliteSessionStore

    store = SqliteSessionStore(db_path, timeout_seconds=3600)
    asyncio.run(store.initialize())
    loaded = asyncio.run(store.get_session("sid-real"))
    assert loaded is not None
    assert loaded.user_id == "u-real"
    assert loaded.token == "tok-real"


def test_migrate_from_json_preserves_document_versions(tmp_path: Path) -> None:
    """End-to-end: every version in the JSON registry lands in SQLite."""
    from datetime import datetime

    from raghub.models import DocumentVersion
    from raghub.storage.json_registry import JsonDocumentRegistry
    from raghub.storage.migration import migrate_from_json

    db_path = tmp_path / "migrated.db"
    registry_path = tmp_path / "registry.json"
    sessions_path = tmp_path / "sessions.json"

    base = datetime(2026, 1, 1, tzinfo=UTC)
    registry = JsonDocumentRegistry(registry_path)
    v1 = DocumentVersion(
        document_id="d-multi",
        version=1,
        checksum="c1",
        owner="a@b.com",
        organization="Acme",
        created_at=base,
        updated_at=base,
    )
    v2 = DocumentVersion(
        document_id="d-multi",
        version=2,
        checksum="c2",
        owner="a@b.com",
        organization="Acme",
        created_at=base,
        updated_at=base,
    )
    registry.save_version(v1)
    registry.save_version(v2)

    sessions_path.write_text(json.dumps({"sessions": {}}), encoding="utf-8")

    asyncio.run(migrate_from_json(str(db_path), str(registry_path), str(sessions_path)))

    from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository

    repo = SqliteDocumentRepository(db_path)
    asyncio.run(repo.initialize())
    versions = asyncio.run(repo.list_versions("d-multi"))
    assert [v.version for v in versions] == [1, 2]
    assert [v.checksum for v in versions] == ["c1", "c2"]
