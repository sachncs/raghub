"""Phase 1.10 — per-user preferences CRUD on :class:`SqliteUserStore`."""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.auth.user_store import SqliteUserStore


@pytest.fixture
async def store(tmp_path: Path) -> SqliteUserStore:
    s = SqliteUserStore(tmp_path / "u.db")
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_user_preferences_table_exists(store: SqliteUserStore) -> None:
    """Initialise creates the ``user_preferences`` table."""
    import aiosqlite

    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'"
        )
        assert await cursor.fetchone() is not None


@pytest.mark.asyncio
async def test_set_and_get_pref(store: SqliteUserStore) -> None:
    """Set + get round-trips a single preference."""
    user = await store.create_user("a@b.c", "pw")
    await store.set_pref(user.user_id, "tool_settings", {"agent_enabled": True})
    value = await store.get_pref(user.user_id, "tool_settings")
    assert value == {"agent_enabled": True}


@pytest.mark.asyncio
async def test_get_prefs_returns_all(store: SqliteUserStore) -> None:
    """``get_prefs`` returns the full mapping for a user."""
    user = await store.create_user("a@b.c", "pw")
    await store.set_pref(user.user_id, "tool_settings", {"a": 1})
    await store.set_pref(user.user_id, "favourite_color", "blue")
    all_prefs = await store.get_prefs(user.user_id)
    assert all_prefs == {"tool_settings": {"a": 1}, "favourite_color": "blue"}


@pytest.mark.asyncio
async def test_set_pref_overwrites(store: SqliteUserStore) -> None:
    """Re-setting the same key replaces the value."""
    user = await store.create_user("a@b.c", "pw")
    await store.set_pref(user.user_id, "tool_settings", {"v": 1})
    await store.set_pref(user.user_id, "tool_settings", {"v": 2})
    assert await store.get_pref(user.user_id, "tool_settings") == {"v": 2}


@pytest.mark.asyncio
async def test_delete_pref(store: SqliteUserStore) -> None:
    """Deleting an existing pref removes it; missing keys are no-ops."""
    user = await store.create_user("a@b.c", "pw")
    await store.set_pref(user.user_id, "k", "v")
    await store.delete_pref(user.user_id, "k")
    assert await store.get_pref(user.user_id, "k") is None
    # Missing key — no exception
    await store.delete_pref(user.user_id, "never_set")


@pytest.mark.asyncio
async def test_set_prefs_bulk(store: SqliteUserStore) -> None:
    """``set_prefs`` upserts multiple keys in one call."""
    user = await store.create_user("a@b.c", "pw")
    await store.set_prefs(user.user_id, {"a": 1, "b": [1, 2, 3]})
    all_prefs = await store.get_prefs(user.user_id)
    assert all_prefs == {"a": 1, "b": [1, 2, 3]}


@pytest.mark.asyncio
async def test_get_prefs_handles_malformed_json(store: SqliteUserStore) -> None:
    """Malformed JSON in the ``value`` column is exposed as raw text."""
    user = await store.create_user("a@b.c", "pw")
    # Bypass the API to inject bad JSON (simulates a legacy/corrupt row).
    import aiosqlite
    import json

    payload = json.dumps({"bad": "value"}).replace("}", "]")  # invalid JSON
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            "INSERT INTO user_preferences (user_id, key, value) VALUES (?, ?, ?)",
            (user.user_id, "broken", payload),
        )
        await db.commit()
    all_prefs = await store.get_prefs(user.user_id)
    assert all_prefs.get("broken") == payload