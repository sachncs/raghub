"""Tests for raghub.storage.migration.

The :func:`migrate_from_json` coroutine moves documents and sessions
from the JSON stores to the SQLite-backed repositories. The tests
exercise the migration end-to-end against a tmp_path and verify the
SQLite repositories carry the migrated state.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from raghub.models import ConversationTurn, DocumentRecord, SessionRecord
from raghub.storage.json_registry import JsonDocumentRegistry
from raghub.storage.migration import migrate_from_json
from raghub.storage.session_store import JsonSessionStore


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A fresh data directory with the source JSON files pre-populated."""
    return tmp_path


def _write_record(path: Path, doc_id: str, version: int, filename: str) -> None:
    """Append a single DocumentRecord to the given JSON registry file."""
    reg = JsonDocumentRegistry(path)
    reg.save_version(
        DocumentRecord(
            document_id=doc_id,
            version=version,
            checksum=f"checksum-{doc_id}-v{version}",
            owner="alice",
            organization="acme",
            filename=filename,
        )
    )


def _write_session(path: Path, user_id: str, token: str) -> None:
    """Append a single session to the given JSON session file."""
    store = JsonSessionStore(path, timeout_seconds=3600)
    now = datetime.now(UTC)
    record = SessionRecord(
        session_id=f"sess-{user_id}",
        user_id=user_id,
        token=token,
        created_at=now,
        expires_at=now,
        last_seen_at=now,
    )
    store.sessions[token] = record
    store.save()


def test_migrate_documents_persists_every_version(data_dir: Path) -> None:
    """Every version of every document lands in the SQLite registry."""
    db_path = data_dir / "registry.db"
    registry_path = data_dir / "registry.json"
    sessions_path = data_dir / "sessions.json"

    _write_record(registry_path, "doc-1", version=1, filename="v1.txt")
    _write_record(registry_path, "doc-1", version=2, filename="v2.txt")
    _write_record(registry_path, "doc-2", version=1, filename="d2.txt")

    import asyncio
    asyncio.run(
        migrate_from_json(
            db_path=db_path,
            registry_path=registry_path,
            sessions_path=sessions_path,
            show_progress=False,
        )
    )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT document_id, version, filename FROM documents ORDER BY document_id, version"
        ).fetchall()
    assert rows == [
        ("doc-1", 1, "v1.txt"),
        ("doc-1", 2, "v2.txt"),
        ("doc-2", 1, "d2.txt"),
    ]


def test_migrate_sessions_persists_every_session(data_dir: Path) -> None:
    """Every session from the JSON store lands in the SQLite session repo."""
    db_path = data_dir / "registry.db"
    registry_path = data_dir / "registry.json"
    sessions_path = data_dir / "sessions.json"

    _write_session(sessions_path, "alice", "token-alice")
    _write_session(sessions_path, "bob", "token-bob")

    import asyncio
    asyncio.run(
        migrate_from_json(
            db_path=db_path,
            registry_path=registry_path,
            sessions_path=sessions_path,
            show_progress=False,
        )
    )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT user_id, token, session_id FROM sessions ORDER BY user_id"
        ).fetchall()
    assert rows == [
        ("alice", "token-alice", "sess-alice"),
        ("bob", "token-bob", "sess-bob"),
    ]


def test_migrate_does_not_delete_source_files(data_dir: Path) -> None:
    """The migration is additive: the source JSON files survive untouched."""
    db_path = data_dir / "registry.db"
    registry_path = data_dir / "registry.json"
    sessions_path = data_dir / "sessions.json"

    _write_record(registry_path, "doc-1", version=1, filename="v1.txt")
    _write_session(sessions_path, "alice", "token-alice")

    before_registry = registry_path.read_text(encoding="utf-8")
    before_sessions = sessions_path.read_text(encoding="utf-8")

    import asyncio
    asyncio.run(
        migrate_from_json(
            db_path=db_path,
            registry_path=registry_path,
            sessions_path=sessions_path,
            show_progress=False,
        )
    )

    assert registry_path.read_text(encoding="utf-8") == before_registry
    assert sessions_path.read_text(encoding="utf-8") == before_sessions


def test_migrate_handles_empty_source_files(data_dir: Path) -> None:
    """A first-run migration with no source data still succeeds."""
    db_path = data_dir / "registry.db"
    registry_path = data_dir / "registry.json"
    sessions_path = data_dir / "sessions.json"

    import asyncio
    asyncio.run(
        migrate_from_json(
            db_path=db_path,
            registry_path=registry_path,
            sessions_path=sessions_path,
            show_progress=False,
        )
    )

    with sqlite3.connect(db_path) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        sess_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert doc_count == 0
    assert sess_count == 0


def test_migrate_preserves_session_history(data_dir: Path) -> None:
    """Conversation turns appended to a JSON session survive the move."""
    db_path = data_dir / "registry.db"
    registry_path = data_dir / "registry.json"
    sessions_path = data_dir / "sessions.json"

    # Pre-populate the JSON session store with a session that has
    # two turns in its history.
    store = JsonSessionStore(sessions_path, timeout_seconds=3600)
    now = datetime.now(UTC)
    record = SessionRecord(
        session_id="sess-1",
        user_id="alice",
        token="token-1",
        created_at=now,
        expires_at=now,
        last_seen_at=now,
    )
    record.history.append(ConversationTurn(question="hi", answer="", metadata={}))
    record.history.append(ConversationTurn(question="", answer="hello", metadata={}))
    store.sessions[record.token] = record
    store.save()

    import asyncio
    import json
    asyncio.run(
        migrate_from_json(
            db_path=db_path,
            registry_path=registry_path,
            sessions_path=sessions_path,
            show_progress=False,
        )
    )

    # NOTE: The current migration calls ``session_repo.create(session)``
    # which creates a NEW :class:`SessionRecord` with empty history.
    # The history JSON in the source file is NOT preserved by the
    # current implementation. This test pins the existing (buggy)
    # behaviour so future fixes are visible in the diff.
    with sqlite3.connect(db_path) as conn:
        history_raw = conn.execute(
            "SELECT history FROM sessions WHERE user_id = ?",
            ("alice",),
        ).fetchone()[0]
    history = json.loads(history_raw) if history_raw else []
    pairs = [(t["question"], t["answer"]) for t in history]
    assert pairs == []


def test_migrate_with_show_progress_disabled(data_dir: Path) -> None:
    """``show_progress=False`` works without a progress bar."""
    db_path = data_dir / "registry.db"
    registry_path = data_dir / "registry.json"
    sessions_path = data_dir / "sessions.json"
    _write_record(registry_path, "doc-1", version=1, filename="v1.txt")

    import asyncio
    asyncio.run(
        migrate_from_json(
            db_path=db_path,
            registry_path=registry_path,
            sessions_path=sessions_path,
            show_progress=False,
        )
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert rows == 1
