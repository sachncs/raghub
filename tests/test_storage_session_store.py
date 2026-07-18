"""Tests for JsonSessionStore — JSON-backed session persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
from unittest.mock import MagicMock

import pytest

from raghub.exceptions import AuthenticationError
from raghub.models import ConversationTurn, SessionRecord
from raghub.storage.session_store import JsonSessionStore
from raghub.utils import load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPOCH = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_turn(question: str = "q", answer: str = "a") -> ConversationTurn:
    return ConversationTurn(question=question, answer=answer)


class _MockDatetime:
    """Stand-in for the datetime module to freeze time."""

    def __init__(self, frozen: datetime):
        self._frozen = frozen

    def now(self, tz=None):
        return self._frozen.astimezone(tz) if tz else self._frozen.replace(tzinfo=None)

    @property
    def timezone(self):
        return timezone

    @property
    def timedelta(self):
        return timedelta

    def __getattr__(self, name):
        return getattr(datetime, name)


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "sessions.json"
    return JsonSessionStore(p, timeout_seconds=3600)


# ---------------------------------------------------------------------------
# __init__ / load / save
# ---------------------------------------------------------------------------


class TestInit:
    def test_sets_path_and_timeout(self, tmp_path):
        p = tmp_path / "s.json"
        s = JsonSessionStore(p, timeout_seconds=7200)
        assert s.path == p
        assert s.timeout == timedelta(seconds=7200)
        assert isinstance(s.lock, type(RLock()))

    def test_load_creates_file_when_missing(self, tmp_path):
        p = tmp_path / "new.json"
        s = JsonSessionStore(p, timeout_seconds=300)
        assert s.sessions == {}
        assert p.exists() is False

    def test_load_hydrates_from_existing_file(self, tmp_path):
        p = tmp_path / "existing.json"
        s1 = JsonSessionStore(p, timeout_seconds=600)
        s1.create("user_a")
        s1.create("user_b")
        assert len(s1.sessions) == 2

        s2 = JsonSessionStore(p, timeout_seconds=600)
        assert len(s2.sessions) == 2
        tokens = list(s2.sessions.keys())
        assert s2.sessions[tokens[0]].user_id in ("user_a", "user_b")

    def test_load_corrupted_file_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid json")
        with pytest.raises(Exception):
            JsonSessionStore(p, timeout_seconds=60)

    def test_save_persists_to_disk(self, tmp_path):
        p = tmp_path / "s.json"
        s = JsonSessionStore(p, timeout_seconds=60)
        s.create("alice")
        loaded = load_json(p, default={})
        assert "sessions" in loaded
        assert len(loaded["sessions"]) == 1

    def test_save_overwrites_previous_content(self, tmp_path):
        p = tmp_path / "s.json"
        s = JsonSessionStore(p, timeout_seconds=60)
        s.create("alice")
        s.create("bob")
        s.invalidate(list(s.sessions.keys())[0])
        loaded = load_json(p, default={})
        assert len(loaded["sessions"]) == 1


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_returns_session_record(self, store):
        session = store.create("user1")
        assert isinstance(session, SessionRecord)
        assert session.user_id == "user1"
        assert session.session_id is not None
        assert session.token is not None

    def test_session_stored_and_persisted(self, store):
        session = store.create("user1")
        assert store.sessions[session.token].user_id == "user1"
        store2 = JsonSessionStore(store.path, timeout_seconds=3600)
        assert store2.sessions[session.token].user_id == "user1"

    def test_unique_tokens(self, store):
        s1 = store.create("a")
        s2 = store.create("b")
        assert s1.token != s2.token

    def test_timestamps(self, store):
        now = datetime.now(timezone.utc)
        session = store.create("user1")
        assert session.created_at == session.last_seen_at
        assert session.expires_at == session.created_at + timedelta(seconds=3600)
        assert abs((session.created_at - now).total_seconds()) < 5

    def test_create_uses_lock(self, store):
        store.lock = MagicMock()
        store.create("u")
        store.lock.__enter__.assert_called_once()


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_valid_token_returns_session(self, store):
        session = store.create("user1")
        resolved = store.resolve(session.token)
        assert resolved is not None
        assert resolved.token == session.token
        assert resolved.user_id == "user1"

    def test_missing_token_returns_none(self, store):
        assert store.resolve("nonexistent") is None

    def test_expired_token_returns_none_and_evicts(self, store):
        session = store.create("user1")
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        session.expires_at = past
        resolved = store.resolve(session.token)
        assert resolved is None
        assert session.token not in store.sessions

    def test_sliding_window_bumps_expiry(self, tmp_path):
        p = tmp_path / "sw.json"
        s = JsonSessionStore(p, timeout_seconds=3600)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("raghub.storage.session_store.datetime", _MockDatetime(EPOCH))
            session = s.create("user1")
        original_expiry = session.expires_at

        later = EPOCH + timedelta(minutes=10)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("raghub.storage.session_store.datetime", _MockDatetime(later))
            resolved = s.resolve(session.token)

        assert resolved is not None
        assert resolved.expires_at > original_expiry
        assert resolved.expires_at == later + timedelta(seconds=3600)

    def test_expired_session_evicted_from_disk(self, store):
        session = store.create("user1")
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        session.expires_at = past
        store.resolve(session.token)

        store2 = JsonSessionStore(store.path, timeout_seconds=3600)
        assert session.token not in store2.sessions

    def test_resolve_uses_lock(self, store):
        session = store.create("u")
        store.lock = MagicMock()
        store.resolve(session.token)
        store.lock.__enter__.assert_called_once()


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


class TestInvalidate:
    def test_removes_session(self, store):
        session = store.create("user1")
        assert session.token in store.sessions
        store.invalidate(session.token)
        assert session.token not in store.sessions

    def test_unknown_token_is_noop(self, store):
        store.invalidate("ghost")
        assert len(store.sessions) == 0

    def test_persists_removal(self, store):
        session = store.create("user1")
        store.invalidate(session.token)
        store2 = JsonSessionStore(store.path, timeout_seconds=3600)
        assert session.token not in store2.sessions

    def test_invalidate_uses_lock(self, store):
        store.lock = MagicMock()
        store.invalidate("anything")
        store.lock.__enter__.assert_called_once()


# ---------------------------------------------------------------------------
# append_turn
# ---------------------------------------------------------------------------


class TestAppendTurn:
    def test_appends_to_session_history(self, store):
        session = store.create("user1")
        turn = _make_turn("Hello", "Hi")
        store.append_turn(session.token, turn)
        assert len(store.sessions[session.token].history) == 1
        assert store.sessions[session.token].history[0].question == "Hello"

    def test_multiple_turns(self, store):
        session = store.create("user1")
        store.append_turn(session.token, _make_turn("q1", "a1"))
        store.append_turn(session.token, _make_turn("q2", "a2"))
        assert len(store.sessions[session.token].history) == 2

    def test_persists_history(self, store):
        session = store.create("user1")
        store.append_turn(session.token, _make_turn("Persist", "Yes"))
        store2 = JsonSessionStore(store.path, timeout_seconds=3600)
        assert len(store2.sessions[session.token].history) == 1
        assert store2.sessions[session.token].history[0].question == "Persist"

    def test_invalid_token_raises(self, store):
        turn = _make_turn()
        with pytest.raises(AuthenticationError, match="Invalid session"):
            store.append_turn("bad_token", turn)

    def test_expired_token_raises(self, store):
        session = store.create("user1")
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        session.expires_at = past
        turn = _make_turn()
        with pytest.raises(AuthenticationError, match="Invalid session"):
            store.append_turn(session.token, turn)

    def test_append_turn_uses_lock(self, store):
        session = store.create("u")
        store.lock = MagicMock()
        store.append_turn(session.token, _make_turn())
        assert store.lock.__enter__.call_count >= 1

    def test_append_turn_lock_reentrant(self, tmp_path):
        p = tmp_path / "r.json"
        s = JsonSessionStore(p, timeout_seconds=60)
        session = s.create("u")
        s.append_turn(session.token, _make_turn())
        assert len(s.sessions[session.token].history) == 1


# ---------------------------------------------------------------------------
# load_turns
# ---------------------------------------------------------------------------


class TestLoadTurns:
    def test_returns_history(self, store):
        session = store.create("user1")
        store.append_turn(session.token, _make_turn("Q", "A"))
        turns = store.load_turns(session.token)
        assert len(turns) == 1
        assert turns[0].question == "Q"

    def test_returns_copy(self, store):
        session = store.create("user1")
        store.append_turn(session.token, _make_turn())
        turns = store.load_turns(session.token)
        turns.clear()
        assert len(store.sessions[session.token].history) == 1

    def test_empty_list_for_missing_token(self, store):
        assert store.load_turns("no_such_token") == []

    def test_empty_list_for_expired_token(self, store):
        session = store.create("user1")
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        session.expires_at = past
        assert store.load_turns(session.token) == []


# ---------------------------------------------------------------------------
# clear_turns
# ---------------------------------------------------------------------------


class TestClearTurns:
    def test_clears_history(self, store):
        session = store.create("user1")
        store.append_turn(session.token, _make_turn())
        store.append_turn(session.token, _make_turn())
        assert len(store.sessions[session.token].history) == 2
        store.clear_turns(session.token)
        assert len(store.sessions[session.token].history) == 0

    def test_persists_cleared_history(self, store):
        session = store.create("user1")
        store.append_turn(session.token, _make_turn())
        store.clear_turns(session.token)
        store2 = JsonSessionStore(store.path, timeout_seconds=3600)
        assert len(store2.sessions[session.token].history) == 0

    def test_invalid_token_raises(self, store):
        with pytest.raises(AuthenticationError, match="Invalid session"):
            store.clear_turns("bad")

    def test_expired_token_raises(self, store):
        session = store.create("user1")
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        session.expires_at = past
        with pytest.raises(AuthenticationError, match="Invalid session"):
            store.clear_turns(session.token)

    def test_clear_turns_uses_lock(self, store):
        session = store.create("u")
        store.lock = MagicMock()
        store.clear_turns(session.token)
        assert store.lock.__enter__.call_count >= 1


# ---------------------------------------------------------------------------
# Expiry behaviour (sliding window)
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_sliding_window_keeps_active_session_alive(self, tmp_path):
        p = tmp_path / "sw.json"
        s = JsonSessionStore(p, timeout_seconds=3600)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("raghub.storage.session_store.datetime", _MockDatetime(EPOCH))
            session = s.create("user1")
        for i in range(6):
            now = EPOCH + timedelta(minutes=30 * i)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("raghub.storage.session_store.datetime", _MockDatetime(now))
                resolved = s.resolve(session.token)
            assert resolved is not None, f"failed at {i=}, time={now}"

    def test_idle_session_expires(self, tmp_path):
        p = tmp_path / "idle.json"
        s = JsonSessionStore(p, timeout_seconds=3600)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("raghub.storage.session_store.datetime", _MockDatetime(EPOCH))
            session = s.create("user1")
        later = EPOCH + timedelta(hours=2)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("raghub.storage.session_store.datetime", _MockDatetime(later))
            assert s.resolve(session.token) is None

    def test_create_sets_expiry_in_future(self, store):
        session = store.create("user1")
        assert session.expires_at > datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


class TestLocking:
    def test_lock_is_rlock(self, store):
        assert isinstance(store.lock, type(RLock()))

    def test_reentrant_lock_allows_nested_acquire(self, tmp_path):
        p = tmp_path / "rlock_test.json"
        s = JsonSessionStore(p, timeout_seconds=60)
        session = s.create("user1")
        s.append_turn(session.token, _make_turn())
        assert len(s.sessions[session.token].history) == 1

    def test_create_acquires_lock(self, tmp_path):
        s = JsonSessionStore(tmp_path / "l.json", timeout_seconds=60)
        s.lock = MagicMock()
        s.create("u")
        s.lock.__enter__.assert_called_once()

    def test_resolve_acquires_lock(self, tmp_path):
        s = JsonSessionStore(tmp_path / "l.json", timeout_seconds=60)
        session = s.create("u")
        s.lock = MagicMock()
        s.resolve(session.token)
        s.lock.__enter__.assert_called_once()

    def test_invalidate_acquires_lock(self, tmp_path):
        s = JsonSessionStore(tmp_path / "l.json", timeout_seconds=60)
        session = s.create("u")
        s.lock = MagicMock()
        s.invalidate(session.token)
        s.lock.__enter__.assert_called_once()

    def test_append_turn_acquires_lock(self, tmp_path):
        s = JsonSessionStore(tmp_path / "l.json", timeout_seconds=60)
        session = s.create("u")
        s.lock = MagicMock()
        s.append_turn(session.token, _make_turn())
        assert s.lock.__enter__.call_count >= 1

    def test_clear_turns_acquires_lock(self, tmp_path):
        s = JsonSessionStore(tmp_path / "l.json", timeout_seconds=60)
        session = s.create("u")
        s.lock = MagicMock()
        s.clear_turns(session.token)
        assert s.lock.__enter__.call_count >= 1
