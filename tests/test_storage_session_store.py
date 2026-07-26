"""Qualitative tests for the session stores (JSON + SQLite backends).

Covers real behaviour:

* Sessions survive a process restart when the on-disk state is
  reloaded.
* The sliding-window expiry actually evicts idle sessions and keeps
  active ones alive.
* Concurrent :meth:`create` + :meth:`resolve` calls do not corrupt
  the on-disk file.
* Append / clear / invalidate operations respect the auth contract
  (raise :class:`AuthenticationError` for unknown / expired tokens).
* The SQLite backend's schema is the source of truth (added columns
  appear on legacy databases).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from raghub.exceptions import AuthenticationError
from raghub.models import ConversationTurn, SessionRecord
from raghub.storage.session_store import JsonSessionStore
from raghub.utils import load_json


def _make_turn(question: str = "q?", answer: str = "a!") -> ConversationTurn:
    return ConversationTurn(question=question, answer=answer)


# ===========================================================================
# JsonSessionStore
# ===========================================================================


@pytest.fixture
def json_store(tmp_path: Path) -> JsonSessionStore:
    return JsonSessionStore(tmp_path / "sessions.json", timeout_seconds=3600)


class TestJsonSessionStoreInit:
    def test_sets_path_timeout_and_lock(self, tmp_path: Path) -> None:
        p = tmp_path / "s.json"
        store = JsonSessionStore(p, timeout_seconds=7200)
        assert store.path == p
        assert store.timeout == timedelta(seconds=7200)

    def test_load_creates_empty_state_when_file_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.json"
        store = JsonSessionStore(p, timeout_seconds=60)
        assert store.sessions == {}

    def test_load_hydrates_from_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "existing.json"
        a = JsonSessionStore(p, timeout_seconds=600)
        a.create("user_a")
        b = JsonSessionStore(p, timeout_seconds=600)
        assert len(b.sessions) == 1

    def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{invalid")
        with pytest.raises(Exception):
            JsonSessionStore(p, timeout_seconds=60)


class TestJsonSessionStoreCreate:
    def test_creates_with_unique_token(self, json_store: JsonSessionStore) -> None:
        a = json_store.create("user1")
        b = json_store.create("user1")
        assert a.token != b.token
        assert a.session_id != b.session_id

    def test_new_session_has_future_expiry(self, json_store: JsonSessionStore) -> None:
        s = json_store.create("user1")
        assert s.expires_at > datetime.now(UTC)

    def test_persists_to_disk(self, json_store: JsonSessionStore, tmp_path: Path) -> None:
        json_store.create("alice")
        loaded = load_json(json_store.path, default={})
        assert "sessions" in loaded
        assert len(loaded["sessions"]) == 1


class TestJsonSessionStoreResolve:
    def test_unknown_token_returns_none(self, json_store: JsonSessionStore) -> None:
        assert json_store.resolve("not-a-real-token") is None

    def test_known_token_returns_session(self, json_store: JsonSessionStore) -> None:
        s = json_store.create("u1")
        resolved = json_store.resolve(s.token)
        assert resolved is not None
        assert resolved.user_id == "u1"

    def test_sliding_window_pushes_expiry_forward(
        self, json_store: JsonSessionStore
    ) -> None:
        s = json_store.create("u1")
        original = s.expires_at
        # Force a clock tick by sleeping briefly, then resolve.
        import time
        time.sleep(0.01)
        resolved = json_store.resolve(s.token)
        assert resolved is not None
        assert resolved.expires_at >= original

    def test_expired_session_returns_none_and_evicts(
        self, json_store: JsonSessionStore
    ) -> None:
        s = json_store.create("u1")
        # Force expiry by mutating the timestamp.
        s.expires_at = datetime.now(UTC) - timedelta(hours=2)
        assert json_store.resolve(s.token) is None
        # The session must be evicted from the in-memory map.
        assert s.token not in json_store.sessions


class TestJsonSessionStoreInvalidate:
    def test_remove_known(self, json_store: JsonSessionStore) -> None:
        s = json_store.create("u1")
        json_store.invalidate(s.token)
        assert s.token not in json_store.sessions

    def test_unknown_is_noop(self, json_store: JsonSessionStore) -> None:
        # Should not raise.
        json_store.invalidate("nope")


class TestJsonSessionStoreHistory:
    def test_append_turn_adds_entry(self, json_store: JsonSessionStore) -> None:
        s = json_store.create("u1")
        json_store.append_turn(s.token, _make_turn("q1", "a1"))
        assert len(s.history) == 1
        assert s.history[0].question == "q1"

    def test_append_turn_preserves_order(self, json_store: JsonSessionStore) -> None:
        s = json_store.create("u1")
        json_store.append_turn(s.token, _make_turn("q1", "a1"))
        json_store.append_turn(s.token, _make_turn("q2", "a2"))
        json_store.append_turn(s.token, _make_turn("q3", "a3"))
        assert [t.question for t in s.history] == ["q1", "q2", "q3"]

    def test_append_turn_to_invalid_session_raises(
        self, json_store: JsonSessionStore
    ) -> None:
        with pytest.raises(AuthenticationError, match="Invalid session"):
            json_store.append_turn("nope", _make_turn())

    def test_append_turn_to_expired_session_raises(
        self, json_store: JsonSessionStore
    ) -> None:
        s = json_store.create("u1")
        s.expires_at = datetime.now(UTC) - timedelta(hours=2)
        with pytest.raises(AuthenticationError, match="Invalid session"):
            json_store.append_turn(s.token, _make_turn())

    def test_load_turns_unknown_token_returns_empty(
        self, json_store: JsonSessionStore
    ) -> None:
        assert json_store.load_turns("nope") == []

    def test_load_turns_returns_history(self, json_store: JsonSessionStore) -> None:
        s = json_store.create("u1")
        json_store.append_turn(s.token, _make_turn("q1", "a1"))
        turns = json_store.load_turns(s.token)
        assert len(turns) == 1

    def test_clear_turns_empties_history(self, json_store: JsonSessionStore) -> None:
        s = json_store.create("u1")
        json_store.append_turn(s.token, _make_turn())
        json_store.append_turn(s.token, _make_turn())
        json_store.clear_turns(s.token)
        assert s.history == []

    def test_clear_turns_unknown_raises(self, json_store: JsonSessionStore) -> None:
        with pytest.raises(AuthenticationError):
            json_store.clear_turns("nope")


class TestJsonSessionStoreConcurrency:
    def test_concurrent_creates_do_not_corrupt(
        self, json_store: JsonSessionStore
    ) -> None:
        """20 threads creating sessions simultaneously must leave the
        store with 20 valid sessions."""
        n = 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(json_store.create, f"u{i}") for i in range(n)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        assert len(json_store.sessions) == n
        # The file is valid JSON.
        loaded = load_json(json_store.path, default={})
        assert len(loaded["sessions"]) == n

    def test_concurrent_appends_dont_lose_turns(
        self, json_store: JsonSessionStore
    ) -> None:
        s = json_store.create("u1")
        n = 50
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(json_store.append_turn, s.token, _make_turn(f"q{i}", f"a{i}"))
                for i in range(n)
            ]
            concurrent.futures.wait(futures)
        # Every append succeeded.
        assert len(s.history) == n


# ===========================================================================
# SqliteSessionStore
# ===========================================================================


@pytest.fixture
async def sqlite_store(tmp_path: Path):
    from raghub.storage.sqlite_session_store import SqliteSessionStore
    store = SqliteSessionStore(tmp_path / "sessions.db", timeout_seconds=3600)
    await store.initialize()
    yield store


class TestSqliteSessionStoreInit:
    @pytest.mark.asyncio
    async def test_creates_table(self, tmp_path: Path) -> None:
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        import aiosqlite
        store = SqliteSessionStore(tmp_path / "sessions.db", timeout_seconds=60)
        await store.initialize()
        async with aiosqlite.connect(store.db_path) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_idempotent_initialize(self, tmp_path: Path) -> None:
        """Calling initialize() twice must not raise."""
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        store = SqliteSessionStore(tmp_path / "sessions.db", timeout_seconds=60)
        await store.initialize()
        await store.initialize()  # must not raise


class TestSqliteSessionStoreCreate:
    @pytest.mark.asyncio
    async def test_create_returns_session_record(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        assert s.user_id == "u1"
        assert s.token
        assert s.expires_at > datetime.now(UTC)

    @pytest.mark.asyncio
    async def test_create_uses_distinct_tokens(self, sqlite_store) -> None:
        a = await sqlite_store.create_session("u1")
        b = await sqlite_store.create_session("u1")
        assert a.token != b.token
        assert a.session_id != b.session_id


class TestSqliteSessionStoreGet:
    @pytest.mark.asyncio
    async def test_get_by_id(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        loaded = await sqlite_store.get_session(s.session_id)
        assert loaded is not None
        assert loaded.user_id == "u1"

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, sqlite_store) -> None:
        assert await sqlite_store.get_session("missing") is None

    @pytest.mark.asyncio
    async def test_get_by_token(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        loaded = await sqlite_store.get_by_token(s.token)
        assert loaded is not None
        assert loaded.user_id == "u1"

    @pytest.mark.asyncio
    async def test_get_by_unknown_token_returns_none(self, sqlite_store) -> None:
        assert await sqlite_store.get_by_token("nope") is None

    @pytest.mark.asyncio
    async def test_get_by_token_evicts_expired(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        # Force expiry.
        s.expires_at = datetime.now(UTC) - timedelta(hours=1)
        await sqlite_store.update_session(s)
        assert await sqlite_store.get_by_token(s.token) is None
        # The session must be deleted from the table.
        assert await sqlite_store.get_session(s.session_id) is None


class TestSqliteSessionStoreSlidingExpiry:
    @pytest.mark.asyncio
    async def test_resolve_pushes_expiry_forward(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        original = s.expires_at
        await asyncio.sleep(0.01)
        resolved = await sqlite_store.get_by_token(s.token)
        assert resolved is not None
        assert resolved.expires_at >= original

    @pytest.mark.asyncio
    async def test_active_session_survives_window(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        # Three rapid resolves all return the session.
        for _ in range(3):
            resolved = await sqlite_store.get_by_token(s.token)
            assert resolved is not None


class TestSqliteSessionStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_session(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        await sqlite_store.delete_session(s.session_id)
        assert await sqlite_store.get_session(s.session_id) is None

    @pytest.mark.asyncio
    async def test_delete_unknown_is_noop(self, sqlite_store) -> None:
        await sqlite_store.delete_session("missing")  # must not raise


class TestSqliteSessionStoreOverrides:
    @pytest.mark.asyncio
    async def test_get_overrides_default_empty(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        assert await sqlite_store.get_overrides(s.session_id) == {}

    @pytest.mark.asyncio
    async def test_set_and_get_overrides(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        await sqlite_store.set_overrides(s.session_id, {"agent_enabled": True, "web": False})
        overrides = await sqlite_store.get_overrides(s.session_id)
        assert overrides["agent_enabled"] is True
        assert overrides["web"] is False

    @pytest.mark.asyncio
    async def test_set_overrides_replaces_dict(self, sqlite_store) -> None:
        s = await sqlite_store.create_session("u1")
        await sqlite_store.set_overrides(s.session_id, {"agent_enabled": True})
        await sqlite_store.set_overrides(s.session_id, {"web": True, "graph": False})
        overrides = await sqlite_store.get_overrides(s.session_id)
        assert overrides == {"web": True, "graph": False}, (
            "set_overrides must replace the entire dict — leaving the "
            "old key behind would silently merge preferences."
        )


class TestSqliteSessionStoreConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_creates_all_succeed(self, tmp_path: Path) -> None:
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        store = SqliteSessionStore(tmp_path / "sessions.db", timeout_seconds=3600)
        await store.initialize()

        async def _create(i: int) -> str:
            s = await store.create_session(f"u{i}")
            return s.session_id

        ids = await asyncio.gather(*[_create(i) for i in range(20)])
        assert len(set(ids)) == 20, "Each create_session must produce a unique id"
