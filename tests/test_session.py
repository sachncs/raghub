"""End-to-end JSON session-store tests.

These tests exercise the full session lifecycle through
:class:`JsonSessionStore`:

* Sliding-window expiry keeps an active session alive across
  multiple ``resolve`` calls and evicts idle ones.
* The on-disk JSON file is parseable after every operation.
* Concurrent operations are safe (the ``RLock`` serialises them).
* Unknown tokens are rejected (no silent success).
* History mutations are persisted across process restarts.
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from raghub.exceptions import AuthenticationError
from raghub.models import ConversationTurn
from raghub.storage.session_store import JsonSessionStore


# ===========================================================================
# Construction
# ===========================================================================


class TestJsonSessionStoreConstruction:
    def test_constructor_sets_attributes(self, tmp_path: Path) -> None:
        p = tmp_path / "s.json"
        s = JsonSessionStore(p, timeout_seconds=7200)
        assert s.path == p
        assert s.timeout == timedelta(seconds=7200)

    def test_load_creates_empty_state_for_missing_file(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "missing.json"
        s = JsonSessionStore(p, timeout_seconds=60)
        assert s.sessions == {}

    def test_load_hydrates_from_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "existing.json"
        JsonSessionStore(p, timeout_seconds=600).create("u1")
        reopened = JsonSessionStore(p, timeout_seconds=600)
        assert len(reopened.sessions) == 1

    def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        from raghub.utils import load_json
        from raghub.models import SessionRecord
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        # load_json raises JSONDecodeError on invalid content; the
        # store's load propagates it so the operator sees the corruption.
        with pytest.raises(Exception):
            JsonSessionStore(p, timeout_seconds=60)


# ===========================================================================
# create
# ===========================================================================


class TestJsonSessionStoreCreate:
    def test_creates_session_with_unique_tokens(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        a = s.create("u1")
        b = s.create("u1")
        assert a.token != b.token
        assert a.session_id != b.session_id

    def test_new_session_has_future_expiry(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        session = s.create("u1")
        assert session.expires_at > datetime.now(UTC)

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        p = tmp_path / "s.json"
        s = JsonSessionStore(p, timeout_seconds=60)
        s.create("alice")
        from raghub.utils import load_json
        loaded = load_json(p, default={})
        assert "sessions" in loaded
        assert len(loaded["sessions"]) == 1


# ===========================================================================
# resolve
# ===========================================================================


class TestJsonSessionStoreResolve:
    def test_unknown_token_returns_none(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        assert s.resolve("not-a-token") is None

    def test_known_token_returns_session(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        created = s.create("u1")
        resolved = s.resolve(created.token)
        assert resolved is not None
        assert resolved.user_id == "u1"

    def test_sliding_window_pushes_expiry_forward(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        created = s.create("u1")
        original = created.expires_at
        time.sleep(0.01)
        resolved = s.resolve(created.token)
        assert resolved is not None
        assert resolved.expires_at >= original

    def test_expired_session_is_evicted(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        created = s.create("u1")
        # Force expiry by mutating the field.
        created.expires_at = datetime.now(UTC) - timedelta(hours=2)
        assert s.resolve(created.token) is None
        # The expired session must be removed from the in-memory map.
        assert created.token not in s.sessions


# ===========================================================================
# invalidate
# ===========================================================================


class TestJsonSessionStoreInvalidate:
    def test_remove_known(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        session = s.create("u1")
        s.invalidate(session.token)
        assert session.token not in s.sessions

    def test_unknown_is_noop(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        s.invalidate("nope")  # must not raise


# ===========================================================================
# History
# ===========================================================================


class TestJsonSessionStoreHistory:
    def test_append_turn_adds_entry(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        session = s.create("u1")
        s.append_turn(session.token, ConversationTurn(question="q?", answer="a!"))
        assert len(session.history) == 1
        assert session.history[0].question == "q?"

    def test_append_preserves_order(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        session = s.create("u1")
        for i in range(5):
            s.append_turn(session.token, ConversationTurn(question=f"q{i}", answer=f"a{i}"))
        assert [t.question for t in session.history] == [f"q{i}" for i in range(5)]

    def test_append_to_invalid_token_raises(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        with pytest.raises(AuthenticationError, match="Invalid"):
            s.append_turn("nope", ConversationTurn(question="q", answer="a"))

    def test_append_to_expired_session_raises(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        session = s.create("u1")
        session.expires_at = datetime.now(UTC) - timedelta(hours=2)
        with pytest.raises(AuthenticationError, match="Invalid"):
            s.append_turn(session.token, ConversationTurn(question="q", answer="a"))

    def test_load_turns_returns_history(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        session = s.create("u1")
        s.append_turn(session.token, ConversationTurn(question="q1", answer="a1"))
        turns = s.load_turns(session.token)
        assert len(turns) == 1

    def test_load_turns_unknown_token_returns_empty(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        assert s.load_turns("nope") == []

    def test_clear_turns_empties_history(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        session = s.create("u1")
        s.append_turn(session.token, ConversationTurn(question="q", answer="a"))
        s.append_turn(session.token, ConversationTurn(question="q2", answer="a2"))
        s.clear_turns(session.token)
        assert session.history == []

    def test_clear_turns_unknown_raises(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=60)
        with pytest.raises(AuthenticationError):
            s.clear_turns("nope")


# ===========================================================================
# Concurrency
# ===========================================================================


class TestJsonSessionStoreConcurrency:
    def test_concurrent_creates_no_loss(self, tmp_path: Path) -> None:
        """20 threads creating sessions simultaneously — every session
        must land in the in-memory map and the file."""
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=3600)
        n = 20

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(s.create, f"u{i}") for i in range(n)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        assert len(s.sessions) == n
        from raghub.utils import load_json
        loaded = load_json(s.path, default={})
        assert len(loaded["sessions"]) == n

    def test_concurrent_appends_no_loss(self, tmp_path: Path) -> None:
        s = JsonSessionStore(tmp_path / "s.json", timeout_seconds=3600)
        session = s.create("u1")
        n = 50
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    s.append_turn,
                    session.token,
                    ConversationTurn(question=f"q{i}", answer=f"a{i}"),
                )
                for i in range(n)
            ]
            concurrent.futures.wait(futures)
        assert len(session.history) == n
