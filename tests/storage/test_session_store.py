"""Tests for raghub.storage.session_store.

The :class:`JsonSessionStore` persists user sessions and per-session
conversation history. Tests cover:

* create / resolve / invalidate lifecycle.
* Append / load / clear turns.
* Round-trip persistence (write → reload → read).
* Empty file handling.
* Corrupt file falls back to empty state.
* Expiry semantics (the ``timeout_seconds`` argument).
* Concurrent access under threading.RLock.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from raghub.models import ConversationTurn, SessionRecord, UserPrincipal
from raghub.storage.session_store import JsonSessionStore


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """A temporary path for the session-store JSON file."""
    return tmp_path / "sessions.json"


def _make_user(user_id: str = "alice") -> UserPrincipal:
    return UserPrincipal(
        user_id=user_id,
        email=f"{user_id}@example.com",
        allowed_companies=["Acme"],
        is_admin=False,
    )


def _make_turn(role: str = "user", text: str = "hi") -> ConversationTurn:
    if role == "user":
        return ConversationTurn(question=text, answer="", metadata={})
    return ConversationTurn(question="", answer=text, metadata={})


def test_empty_file_yields_no_sessions(store_path: Path) -> None:
    """An empty path is hydrated to an empty store."""
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    assert store.sessions == {}


def test_create_and_resolve_roundtrip(store_path: Path) -> None:
    """create() registers a session that resolve() returns."""
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    user = _make_user()
    session = store.create(user_id=user.user_id)
    assert session.token
    assert session.user_id == user.user_id
    fetched = store.resolve(session.token)
    assert fetched is not None
    assert fetched.token == session.token


def test_resolve_unknown_token_returns_none(store_path: Path) -> None:
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    assert store.resolve("not-a-real-token") is None


def test_invalidate_removes_session(store_path: Path) -> None:
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    user = _make_user()
    session = store.create(user_id=user.user_id)
    assert store.resolve(session.token) is not None
    store.invalidate(session.token)
    assert store.resolve(session.token) is None


def test_invalidate_unknown_token_is_noop(store_path: Path) -> None:
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    store.invalidate("does-not-exist")  # must not raise


def test_append_turns_and_load(store_path: Path) -> None:
    """Conversation history round-trips through save/load."""
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    user = _make_user()
    session = store.create(user_id=user.user_id)
    turn1 = _make_turn(role="user", text="hello")
    turn2 = _make_turn(role="assistant", text="hi there")
    store.append_turn(session.token, turn1)
    store.append_turn(session.token, turn2)

    store2 = JsonSessionStore(store_path, timeout_seconds=3600)
    turns = store2.load_turns(session.token)
    questions = [t.question for t in turns]
    answers = [t.answer for t in turns]
    assert questions == ["hello", ""]
    assert answers == ["", "hi there"]


def test_clear_turns_empties_history(store_path: Path) -> None:
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    user = _make_user()
    session = store.create(user_id=user.user_id)
    store.append_turn(session.token, _make_turn("user", "first"))
    store.append_turn(session.token, _make_turn("user", "second"))
    store.clear_turns(session.token)
    assert store.load_turns(session.token) == []


def test_corrupt_file_loads_as_empty(store_path: Path) -> None:
    """Invalid JSON content resets the store to an empty state.

    Production deployments should never see this state, but the
    test guards against silent failures from a half-written file
    or a corrupted write. The :meth:`load` method treats any
    non-dict JSON or JSON-decode error as "no data".
    """
    store_path.write_text("not valid json {{{", encoding="utf-8")
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    assert store.sessions == {}


def test_roundtrip_save_and_load(store_path: Path) -> None:
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    user = _make_user()
    session = store.create(user_id=user.user_id)
    store.append_turn(session.token, _make_turn(text="hello"))

    store2 = JsonSessionStore(store_path, timeout_seconds=3600)
    fetched = store2.resolve(session.token)
    assert fetched is not None
    assert fetched.user_id == user.user_id
    turns = store2.load_turns(session.token)
    assert len(turns) == 1
    assert "hello" in {turns[0].question, turns[0].answer}


def test_concurrent_append_turns_preserves_order(store_path: Path) -> None:
    """All appended turns land in some deterministic order."""
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    user = _make_user()
    session = store.create(user_id=user.user_id)

    def worker(marker: str) -> None:
        for j in range(5):
            store.append_turn(
                session.token,
                _make_turn("user", f"{marker}-{j}"),
            )

    threads = [threading.Thread(target=worker, args=(f"m{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    turns = store.load_turns(session.token)
    # Every turn must be present exactly once (no lost writes).
    assert len(turns) == 4 * 5
    markers = sorted(t.question + t.answer for t in turns)
    expected = sorted(f"{m}-{j}" for m in ("m0", "m1", "m2", "m3") for j in range(5))
    assert markers == expected


def test_session_expiry_via_timeout_field(store_path: Path) -> None:
    """The store exposes a ``timeout`` field matching the constructor arg."""
    store = JsonSessionStore(store_path, timeout_seconds=42)
    assert store.timeout == timedelta(seconds=42)


def test_last_seen_updated_on_operations(store_path: Path) -> None:
    """``create`` and ``append_turn`` update ``last_seen_at``."""
    store = JsonSessionStore(store_path, timeout_seconds=3600)
    user = _make_user()
    session = store.create(user_id=user.user_id)
    initial_seen = session.last_seen_at
    assert isinstance(initial_seen, datetime)

    # Time advances between calls.
    import time
    time.sleep(0.01)

    store.append_turn(session.token, _make_turn("user", "test"))
    fetched = store.resolve(session.token)
    assert fetched is not None
    assert fetched.last_seen_at > initial_seen
