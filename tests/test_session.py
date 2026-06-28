"""Tests for session store edge cases."""

from __future__ import annotations

import asyncio

import pytest

from raghub.models import ConversationTurn
from raghub.storage.database import DatabaseManager
from raghub.storage.sqlite_session_store import SqliteSessionStore

pytestmark = pytest.mark.asyncio


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_sessions.db")


async def _make_store(db_path: str) -> SqliteSessionStore:
    store = SqliteSessionStore(db_path, timeout_seconds=60)
    await store.initialize()
    return store


async def test_create_and_get_by_token(tmp_db: str) -> None:
    store = await _make_store(tmp_db)
    session = await store.create_session("user1")
    assert session.token is not None
    loaded = await store.get_by_token(session.token)
    assert loaded is not None
    assert loaded.user_id == "user1"
    assert loaded.session_id == session.session_id


async def test_get_by_token_expired(tmp_db: str) -> None:
    store = SqliteSessionStore(tmp_db, timeout_seconds=0)
    await store.initialize()
    session = await store.create_session("user1")
    await asyncio.sleep(0.01)
    loaded = await store.get_by_token(session.token)
    assert loaded is None


async def test_get_by_token_nonexistent(tmp_db: str) -> None:
    store = await _make_store(tmp_db)
    loaded = await store.get_by_token("nonexistent-token")
    assert loaded is None


async def test_append_and_get_history(tmp_db: str) -> None:
    store = await _make_store(tmp_db)
    session = await store.create_session("user1")
    turn = ConversationTurn(question="Hello", answer="Hi there!")
    await store.append_history(session.session_id, turn)
    history = await store.get_history(session.session_id)
    assert len(history) == 1
    assert history[0].question == "Hello"
    assert history[0].answer == "Hi there!"


async def test_append_history_nonexistent_session(tmp_db: str) -> None:
    store = await _make_store(tmp_db)
    turn = ConversationTurn(question="Hello", answer="Hi")
    await store.append_history("nonexistent", turn)
    history = await store.get_history("nonexistent")
    assert history == []


async def test_get_by_token_updates_expiry(tmp_db: str) -> None:
    store = SqliteSessionStore(tmp_db, timeout_seconds=3600)
    await store.initialize()
    session = await store.create_session("user1")
    original_expiry = session.expires_at
    loaded = await store.get_by_token(session.token)
    assert loaded is not None
    assert loaded.last_seen_at > session.last_seen_at
    assert loaded.expires_at > original_expiry


async def test_delete_session(tmp_db: str) -> None:
    store = await _make_store(tmp_db)
    session = await store.create_session("user1")
    await store.delete_session(session.session_id)
    loaded = await store.get_session(session.session_id)
    assert loaded is None


async def test_multiple_sessions_same_user(tmp_db: str) -> None:
    store = await _make_store(tmp_db)
    s1 = await store.create_session("user1")
    s2 = await store.create_session("user1")
    assert s1.session_id != s2.session_id
    assert s1.token != s2.token
    loaded1 = await store.get_by_token(s1.token)
    loaded2 = await store.get_by_token(s2.token)
    assert loaded1 is not None
    assert loaded2 is not None
    assert loaded1.user_id == "user1"
    assert loaded2.user_id == "user1"


async def test_database_manager_shared_connection(tmp_db: str) -> None:
    db_manager = DatabaseManager(tmp_db)
    await db_manager.connect()
    store = SqliteSessionStore(tmp_db, db_manager=db_manager)
    await store.initialize()
    session = await store.create_session("user1")
    loaded = await store.get_by_token(session.token)
    assert loaded is not None
    assert loaded.user_id == "user1"
    await db_manager.close()
