from __future__ import annotations

import json
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
        patch("raghub.storage.json_registry.JsonDocumentRegistry") as mock_json_reg_cls,
        patch("raghub.storage.session_store.JsonSessionStore") as mock_json_ss_cls,
        patch("raghub.storage.sqlite_session_store.SqliteSessionStore") as mock_sql_ss_cls,
    ):
        mock_repo = AsyncMock()
        mock_repo_cls.return_value = mock_repo
        mock_json_reg = MagicMock()
        mock_json_reg.documents = {}
        mock_json_reg_cls.return_value = mock_json_reg
        mock_sql_ss = AsyncMock()
        mock_sql_ss_cls.return_value = mock_sql_ss
        mock_json_ss = MagicMock()
        mock_json_ss.sessions = {}
        mock_json_ss_cls.return_value = mock_json_ss

        from raghub.storage.migration import migrate_from_json

        import asyncio

        asyncio.run(migrate_from_json(str(db_path), str(registry_path), str(sessions_path)))
        mock_repo.initialize.assert_awaited_once()
        mock_sql_ss.initialize.assert_awaited_once()


def test_migrate_from_json_with_data(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    registry_path = tmp_path / "registry.json"
    sessions_path = tmp_path / "sessions.json"

    registry_path.write_text(json.dumps({"documents": {}}), encoding="utf-8")
    sessions_path.write_text(json.dumps({"sessions": {}}), encoding="utf-8")

    with (
        patch("raghub.repositories.sqlite_document_repo.SqliteDocumentRepository") as mock_repo_cls,
        patch("raghub.storage.json_registry.JsonDocumentRegistry") as mock_json_reg_cls,
        patch("raghub.storage.session_store.JsonSessionStore") as mock_json_ss_cls,
        patch("raghub.storage.sqlite_session_store.SqliteSessionStore") as mock_sql_ss_cls,
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
        mock_sql_ss = AsyncMock()
        mock_sql_ss_cls.return_value = mock_sql_ss
        mock_session = MagicMock(session_id="s1", user_id="u1", token="tok1")
        mock_json_ss = MagicMock()
        mock_json_ss.sessions = {"s1": mock_session}
        mock_json_ss_cls.return_value = mock_json_ss

        from raghub.storage.migration import migrate_from_json

        import asyncio

        asyncio.run(migrate_from_json(str(db_path), str(registry_path), str(sessions_path)))
        mock_repo.initialize.assert_awaited_once()
        mock_sql_ss.initialize.assert_awaited_once()
        mock_repo.save.assert_awaited_once()
        mock_sql_ss.update_session.assert_awaited_once_with(mock_session)
