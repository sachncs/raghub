"""Phase 1.9 + 1.12 — ``overrides`` column on the SQLite session store."""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.storage.sqlite_session_store import SqliteSessionStore


@pytest.fixture
async def store(tmp_path: Path) -> SqliteSessionStore:
    s = SqliteSessionStore(tmp_path / "s.db")
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_overrides_default_empty(store: SqliteSessionStore) -> None:
    session = await store.create_session("user-1")
    assert session.overrides == {}


@pytest.mark.asyncio
async def test_set_overrides_persists(store: SqliteSessionStore) -> None:
    session = await store.create_session("user-1")
    await store.set_overrides(session.session_id, {"agent_enabled": True})
    loaded = await store.get_by_token(session.token)
    assert loaded is not None
    assert loaded.overrides == {"agent_enabled": True}


@pytest.mark.asyncio
async def test_re_initialize_is_idempotent(store: SqliteSessionStore) -> None:
    """Re-running ``initialize`` does not blow up on the ALTER TABLE."""
    session = await store.create_session("user-1")
    await store.set_overrides(session.session_id, {"k": "v"})
    await store.initialize()  # second call
    loaded = await store.get_by_token(session.token)
    assert loaded is not None
    assert loaded.overrides == {"k": "v"}


@pytest.mark.asyncio
async def test_legacy_row_without_overrides_column_loads(tmp_path: Path) -> None:
    """A row written before the migration has no ``overrides`` column.

    The store must default to ``{}`` instead of raising.
    """
    import aiosqlite
    from datetime import UTC, datetime, timedelta

    db_path = tmp_path / "legacy.db"
    # Create the legacy schema (no overrides column).
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                history TEXT DEFAULT '[]'
            );
        """)
        now = datetime.now(UTC)
        await db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s-1", "u-1", "t-1", now.isoformat(), (now + timedelta(hours=1)).isoformat(), now.isoformat(), "[]"),
        )
        await db.commit()

    store = SqliteSessionStore(db_path)
    await store.initialize()  # runs the ALTER TABLE
    session = await store.get_by_token("t-1")
    assert session is not None
    assert session.overrides == {}