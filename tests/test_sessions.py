"""SQLite Sessions store coverage tests.

Exercises the async aiosqlite-backed :class:`Sessions` class and the
related lifespan helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from raghub.models import Session
from raghub.stores import Sessions


@pytest.fixture
async def sqlite_sessions(tmp_path):
    """A Sessions store backed by a temporary SQLite database."""

    sessions = Sessions(str(tmp_path / "sessions.db"), timeout_seconds=3600)
    await sessions.initialize()
    yield sessions
    # No explicit close method; the per-call connection is committed and
    # closed inside maybe_commit_close.


def _make_session(user_id: str = "u1") -> Session:
    """Build a minimal Session fixture."""

    now = datetime.now(UTC)
    return Session(
        id="s-id",
        user_id=user_id,
        token="tok-1",
        created_at=now,
        expires_at=now + timedelta(seconds=3600),
        last_seen_at=now,
    )


@pytest.mark.asyncio
async def test_sessions_create_and_get_by_token(sqlite_sessions: Sessions) -> None:
    """Sessions.create_session stores and get_by_token retrieves."""

    session = await sqlite_sessions.create_session("u1")
    found = await sqlite_sessions.get_by_token(session.token)
    assert found is not None
    assert found.user_id == "u1"


@pytest.mark.asyncio
async def test_sessions_get_by_token_unknown(sqlite_sessions: Sessions) -> None:
    """get_by_token returns None for an unknown token."""

    assert await sqlite_sessions.get_by_token("nope") is None


@pytest.mark.asyncio
async def test_sessions_create_unique_ids(sqlite_sessions: Sessions) -> None:
    """Each create_session call returns a session with a unique id."""

    s1 = await sqlite_sessions.create_session("u1")
    s2 = await sqlite_sessions.create_session("u2")
    assert s1.id != s2.id
    assert s1.user_id == "u1"
    assert s2.user_id == "u2"


@pytest.mark.asyncio
async def test_sessions_delete_session(sqlite_sessions: Sessions) -> None:
    """delete_session removes by session_id."""

    session = await sqlite_sessions.create_session("u1")
    await sqlite_sessions.delete_session(session.id)
    assert await sqlite_sessions.get_by_token(session.token) is None


@pytest.mark.asyncio
async def test_sessions_delete_session_missing_noop(sqlite_sessions: Sessions) -> None:
    """delete_session on an unknown id is a no-op."""

    await sqlite_sessions.delete_session("missing")  # does not raise


@pytest.mark.asyncio
async def test_sessions_load_after_close_round_trip(tmp_path) -> None:
    """A session created in one instance is readable in a fresh instance."""

    path = str(tmp_path / "rt.db")
    first = Sessions(path, timeout_seconds=3600)
    await first.initialize()
    session = await first.create_session("u1")
    found = await first.get_by_token(session.token)
    assert found is not None

    second = Sessions(path, timeout_seconds=3600)
    await second.initialize()
    loaded = await second.get_by_token(session.token)
    assert loaded is not None


@pytest.mark.asyncio
async def test_sessions_get_session_by_id(sqlite_sessions: Sessions) -> None:
    """get_session finds the session by its session_id."""

    session = await sqlite_sessions.create_session("u1")
    found = await sqlite_sessions.get_session(session.id)
    assert found is not None
    assert found.id == session.id


@pytest.mark.asyncio
async def test_sessions_get_session_unknown(sqlite_sessions: Sessions) -> None:
    """get_session returns None for an unknown id."""

    assert await sqlite_sessions.get_session("missing") is None


@pytest.mark.asyncio
async def test_sessions_create_session_record(sqlite_sessions: Sessions) -> None:
    """create_session_record stores a fully-built Session."""

    session = Session(
        id="manual-id",
        user_id="u1",
        token="manual-token",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(seconds=3600),
        last_seen_at=datetime.now(UTC),
    )
    await sqlite_sessions.create_session_record(session)
    found = await sqlite_sessions.get_by_token("manual-token")
    assert found is not None
    assert found.id == "manual-id"


@pytest.mark.asyncio
async def test_sessions_json_factory(tmp_path) -> None:
    """Sessions.json returns a JsonSessions with the same path / timeout."""

    from raghub.stores import JsonSessions

    path = tmp_path / "sessions.json"
    store = Sessions.json(path, timeout_seconds=3600)
    assert isinstance(store, JsonSessions)
    assert store.timeout == timedelta(seconds=3600)
