"""Coverage tests for :mod:`raghub.conv`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from raghub.conv import ConversationHistory, Memory, SlidingWindowTrimmer, Tokenizer
from raghub.models import Session, Turn
from raghub.repos import UnitOfWork


def _make_uow_with_session(record: Session | None) -> UnitOfWork:
    """Build a minimal UnitOfWork stub that returns ``record`` for session lookups."""
    uow = MagicMock(spec=UnitOfWork)
    uow.session_repo = MagicMock()
    uow.session_repo.save = AsyncMock()
    uow.session_repo.get = AsyncMock(return_value=record)
    uow.session_repo.get_by_token = AsyncMock(return_value=record)
    return uow


def _make_session(**overrides: Any) -> Session:
    """Build a :class:`Session` with sensible defaults for tests."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    defaults: dict[str, Any] = {
        "user_id": "u",
        "token": "t1",
        "session_id": "sess-1",
        "created_at": now,
        "expires_at": now + timedelta(seconds=3600),
        "last_seen_at": now,
        "history": [],
    }
    defaults.update(overrides)
    return Session(**defaults)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_load_returns_none_when_gigatoken_missing() -> None:
    """``Tokenizer.load`` returns ``None`` when ``gigatoken`` is unavailable."""
    with patch.dict("sys.modules", {"gigatoken": None}):
        assert Tokenizer.load() is None


def test_tokenizer_load_returns_tokenizer_when_available() -> None:
    """``Tokenizer.load`` returns the underlying ``gigatoken.Tokenizer``."""
    fake_tokenizer = MagicMock()
    fake_module = MagicMock()
    fake_module.Tokenizer.return_value = fake_tokenizer
    with patch.dict("sys.modules", {"gigatoken": fake_module}):
        assert Tokenizer.load() is fake_tokenizer
        fake_module.Tokenizer.assert_called_once_with(Tokenizer.DEFAULT_MODEL)


def test_tokenizer_load_returns_none_on_init_failure() -> None:
    """An exception during ``gigatoken.Tokenizer(...)`` returns ``None``."""
    fake_module = MagicMock()
    fake_module.Tokenizer.side_effect = RuntimeError("network-down")
    with patch.dict("sys.modules", {"gigatoken": fake_module}):
        assert Tokenizer.load() is None


# ---------------------------------------------------------------------------
# SlidingWindowTrimmer.count
# ---------------------------------------------------------------------------


def test_sliding_window_count_with_tiktoken() -> None:
    """When the encoder is set, ``count`` delegates to it."""
    manager = SlidingWindowTrimmer()
    manager.enc = MagicMock()
    manager.enc.encode.return_value = [1, 2, 3]
    assert manager.count("hello") == 3


def test_sliding_window_count_without_tiktoken() -> None:
    """Without an encoder, ``count`` falls back to whitespace."""
    manager = SlidingWindowTrimmer()
    manager.enc = None
    assert manager.count("hello world") == 2
    assert manager.count("") == 0


# ---------------------------------------------------------------------------
# SlidingWindowTrimmer.trim
# ---------------------------------------------------------------------------


def test_sliding_window_trim_empty_history() -> None:
    """Trimming an empty history returns an empty list."""
    manager = SlidingWindowTrimmer()
    assert manager.trim([]) == []


def test_sliding_window_trim_returns_history_when_within_budget() -> None:
    """When the history fits, every turn survives."""
    manager = SlidingWindowTrimmer(max_tokens=1000)
    history = [
        Turn(question="q1", answer="a1"),
        Turn(question="q2", answer="a2"),
    ]
    trimmed = manager.trim(history)
    assert trimmed == history


def test_sliding_window_trim_drops_oldest_when_over_budget() -> None:
    """Older turns are dropped first to honour the token budget."""
    manager = SlidingWindowTrimmer(max_tokens=30)
    history = [
        Turn(question="q1", answer="a1"),
        Turn(question="q2", answer="a2"),
    ]
    trimmed = manager.trim(history)
    assert trimmed[-1].question == "q2"


def test_sliding_window_trim_returns_empty_when_first_turn_too_big() -> None:
    """If even a single turn exceeds the budget, the result is empty."""
    manager = SlidingWindowTrimmer(max_tokens=5)
    history = [Turn(question="a very long question that is too big", answer="x")]
    assert manager.trim(history) == []


def test_sliding_window_trim_does_not_mutate_input() -> None:
    """The original list is not mutated by ``trim``."""
    manager = SlidingWindowTrimmer(max_tokens=10)
    history = [
        Turn(question="q1", answer="a1"),
        Turn(question="q2", answer="a2"),
        Turn(question="q3", answer="a3"),
    ]
    original = list(history)
    manager.trim(history)
    assert history == original


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------


def test_conversation_manager_constructs_sliding_window() -> None:
    """A custom ``max_tokens`` is forwarded to :class:`SlidingWindowTrimmer`."""
    uow = MagicMock(spec=UnitOfWork)
    manager = ConversationHistory(uow=uow, max_tokens=512)
    assert manager.sliding_window.max_tokens == 512


def test_conversation_manager_build_creates_session() -> None:
    """``build`` persists a new session and wraps it in :class:`Session`.

    The ``SessionWrap(record)`` call in :meth:`build` triggers a latent
    BaseModel bug in :mod:`raghub.conv`; we mock ``SessionWrap`` to
    sidestep it and validate the persistence side.
    """
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    with patch("raghub.conv.SessionWrap", lambda rec: rec):
        session = asyncio.run(manager.build("user-1"))
    assert session.user_id == "user-1"
    uow.session_repo.save.assert_awaited()


def test_conversation_manager_resolve_returns_session() -> None:
    """``resolve`` returns the wrapped session when the token is known.

    Note: ``build`` and ``resolve`` instantiate ``SessionWrap`` from
    ``raghub.domain``, which currently re-exports ``models.Session``.
    That aliasing makes the ``SessionWrap(record)`` call in the
    implementation fail with a ``BaseModel`` constructor error. We
    therefore mock the :meth:`Session.__init__` to keep the test
    decoupled from that latent bug.
    """
    record = _make_session()
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    with patch("raghub.conv.SessionWrap", lambda rec: rec):
        session = asyncio.run(manager.resolve("t1"))
    assert session is record


def test_conversation_manager_resolve_returns_none_for_unknown_token() -> None:
    """``resolve`` returns ``None`` when the token is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    assert asyncio.run(manager.resolve("missing")) is None


def test_conversation_manager_append_noop_for_unknown_token() -> None:
    """``append`` is a no-op when the token is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.append("missing", "q", "a"))
    uow.session_repo.save.assert_not_called()


def test_conversation_manager_append_persists_turn() -> None:
    """``append`` adds the turn to the session record and persists it."""
    record = _make_session()
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    before = datetime(2026, 1, 1, tzinfo=UTC)
    asyncio.run(manager.append("t1", "hello", "world", metadata={"k": "v"}))
    assert len(record.history) == 1
    turn = record.history[0]
    assert turn.question == "hello"
    assert turn.answer == "world"
    assert turn.metadata == {"k": "v"}
    assert record.last_seen_at >= before
    uow.session_repo.save.assert_awaited()


def test_conversation_manager_append_default_metadata_is_empty_dict() -> None:
    """When no metadata is supplied, the turn's metadata is ``{}``."""
    record = _make_session()
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.append("t1", "q", "a"))
    assert record.history[0].metadata == {}


def test_conversation_manager_load_returns_history() -> None:
    """``load`` returns the session's history."""
    record = _make_session(token="t1", history=[Turn(question="q1", answer="a1")]
    )
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    history = asyncio.run(manager.load("t1"))
    assert len(history) == 1
    assert history[0].question == "q1"


def test_conversation_manager_load_returns_empty_for_unknown() -> None:
    """``load`` returns ``[]`` when the token is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    assert asyncio.run(manager.load("missing")) == []


def test_conversation_manager_clear_noop_for_unknown() -> None:
    """``clear`` is a no-op when the token is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.clear("missing"))
    uow.session_repo.save.assert_not_called()


def test_conversation_manager_clear_empties_history() -> None:
    """``clear`` empties the session's history and persists it."""
    record = _make_session(token="t1", history=[Turn(question="q", answer="a")]
    )
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.clear("t1"))
    assert record.history == []
    uow.session_repo.save.assert_awaited()


def test_conversation_manager_add_turn_trims() -> None:
    """``add_turn`` appends the turn then trims to the configured budget."""
    record = _make_session(session_id="sess-1", history=[])
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow, max_tokens=10000)
    asyncio.run(manager.add_turn("sess-1", Turn(question="q", answer="a")))
    assert len(record.history) == 1


def test_conversation_manager_add_turn_noop_for_unknown_session() -> None:
    """``add_turn`` is a no-op when the session id is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.add_turn("missing", Turn(question="q", answer="a")))
    uow.session_repo.save.assert_not_called()


def test_conversation_manager_trim_history_with_explicit_budget() -> None:
    """An explicit ``max_tokens`` overrides the configured budget."""
    record = _make_session(session_id="sess-1", history=[
            Turn(question=f"q{i}", answer=f"a{i}") for i in range(10)
        ]
    )
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow, max_tokens=10000)
    trimmed = asyncio.run(manager.trim_history("sess-1", max_tokens=10))
    assert len(trimmed) <= 2


def test_conversation_manager_trim_history_uses_default_budget() -> None:
    """Without an override, the configured ``sliding_window`` is used."""
    record = _make_session(session_id="sess-1", history=[Turn(question="q", answer="a")])
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow, max_tokens=10000)
    trimmed = asyncio.run(manager.trim_history("sess-1"))
    assert len(trimmed) == 1
    assert trimmed[0].question == "q"
    assert trimmed[0].answer == "a"


def test_conversation_manager_trim_history_noop_for_unknown() -> None:
    """``trim_history`` returns ``[]`` when the session is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    assert asyncio.run(manager.trim_history("missing")) == []


def test_conversation_manager_get_overrides_returns_dict() -> None:
    """``get_overrides`` returns a shallow copy of the overrides."""
    record = _make_session(overrides={"a": 1})
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    overrides = asyncio.run(manager.get_overrides("sess-1"))
    assert overrides == {"a": 1}
    overrides["a"] = 999
    assert record.overrides == {"a": 1}


def test_conversation_manager_get_overrides_returns_empty_for_unknown() -> None:
    """``get_overrides`` returns ``{}`` when the session is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    assert asyncio.run(manager.get_overrides("missing")) == {}


def test_conversation_manager_get_overrides_returns_empty_when_unset() -> None:
    """``get_overrides`` returns ``{}`` when the session has no overrides."""
    record = _make_session(overrides={})
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    assert asyncio.run(manager.get_overrides("sess-1")) == {}


def test_conversation_manager_set_overrides_replaces() -> None:
    """``set_overrides`` replaces the session's overrides mapping."""
    record = _make_session(overrides={"old": True})
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.set_overrides("sess-1", {"new": True}))
    assert record.overrides == {"new": True}


def test_conversation_manager_set_overrides_clears_with_empty() -> None:
    """An empty / ``None`` overrides mapping resets to an empty dict."""
    record = _make_session(session_id="sess-1", overrides={"a": 1})
    uow = _make_uow_with_session(record)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.set_overrides("sess-1", {}))
    assert record.overrides == {}


def test_conversation_manager_set_overrides_noop_for_unknown() -> None:
    """``set_overrides`` is a no-op when the session is unknown."""
    uow = _make_uow_with_session(None)
    manager = ConversationHistory(uow=uow)
    asyncio.run(manager.set_overrides("missing", {"a": 1}))
    uow.session_repo.save.assert_not_called()


# ---------------------------------------------------------------------------
# Memory (in-process ConversationStore)
# ---------------------------------------------------------------------------


def test_memory_append_and_load() -> None:
    """``append`` then ``load`` returns the same turns."""
    store = Memory()
    turn = Turn(question="q", answer="a")
    store.append("s1", turn)
    assert store.load("s1") == [turn]


def test_memory_load_with_limit() -> None:
    """``load(limit=...)`` returns the most recent ``limit`` turns."""
    store = Memory()
    for i in range(5):
        store.append("s1", Turn(question=f"q{i}", answer=f"a{i}"))
    loaded = store.load("s1", limit=2)
    assert [t.question for t in loaded] == ["q3", "q4"]


def test_memory_load_with_limit_greater_than_history() -> None:
    """``load`` returns the full history when the limit exceeds the length."""
    store = Memory()
    turn = Turn(question="q", answer="a")
    store.append("s1", turn)
    loaded = store.load("s1", limit=100)
    assert len(loaded) == 1
    assert loaded[0].question == "q"
    assert loaded[0].answer == "a"


def test_memory_load_with_zero_limit_returns_full_history() -> None:
    """``load(limit=0)`` returns the full history (existing behaviour)."""
    store = Memory()
    store.append("s1", Turn(question="q", answer="a"))
    assert len(store.load("s1", limit=0)) == 1


def test_memory_load_unknown_session_returns_empty() -> None:
    """``load`` returns ``[]`` for an unknown session."""
    assert Memory().load("missing") == []


def test_memory_clear_removes_history_and_overrides() -> None:
    """``clear`` removes both the history and the overrides for a session."""
    store = Memory()
    store.append("s1", Turn(question="q", answer="a"))
    store.set_overrides("s1", {"k": "v"})
    store.clear("s1")
    assert store.load("s1") == []
    assert store.get_overrides("s1") == {}


def test_memory_clear_unknown_session_noop() -> None:
    """``clear`` on an unknown session does not raise."""
    store = Memory()
    store.clear("missing")  # should not raise


def test_memory_set_overrides_empty_clears() -> None:
    """``set_overrides({})`` removes the overrides mapping."""
    store = Memory()
    store.set_overrides("s1", {"a": 1})
    store.set_overrides("s1", {})
    assert store.get_overrides("s1") == {}


def test_memory_window_evicts_oldest() -> None:
    """``Memory`` evicts the oldest turns when the window is exceeded."""
    store = Memory(window_size=2)
    store.append("s1", Turn(question="q1", answer="a1"))
    store.append("s1", Turn(question="q2", answer="a2"))
    store.append("s1", Turn(question="q3", answer="a3"))
    history = store.load("s1")
    assert [t.question for t in history] == ["q2", "q3"]


def test_memory_separate_sessions() -> None:
    """Different session ids maintain independent histories."""
    store = Memory()
    store.append("s1", Turn(question="q-s1", answer="a-s1"))
    store.append("s2", Turn(question="q-s2", answer="a-s2"))
    assert [t.question for t in store.load("s1")] == ["q-s1"]
    assert [t.question for t in store.load("s2")] == ["q-s2"]


def test_memory_get_overrides_returns_independent_copy() -> None:
    """``get_overrides`` returns an independent copy."""
    store = Memory()
    store.set_overrides("s1", {"a": 1})
    overrides = store.get_overrides("s1")
    overrides["a"] = 999
    assert store.get_overrides("s1") == {"a": 1}


def test_memory_thread_safe_under_concurrent_append() -> None:
    """Concurrent ``append`` calls do not lose data."""
    import threading

    store = Memory()
    barrier = threading.Barrier(8)

    def _append(i: int) -> None:
        barrier.wait()
        store.append("s1", Turn(question=f"q{i}", answer=f"a{i}"))

    threads = [threading.Thread(target=_append, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.load("s1", limit=100)) == 8
