"""Tests for ``raghub.pipeline.router`` (Router facade)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from raghub.pipeline.router import Router
from raghub.models import Turn


def test_router_load_history_returns_empty_when_session_id_none() -> None:
    """``Router.load_history(None)`` returns [] without touching the store."""

    store = MagicMock()
    router = Router(store)
    assert router.load_history(None) == []
    store.load.assert_not_called()


def test_router_load_history_delegates_to_store() -> None:
    """``Router.load_history`` returns the store's load() result."""

    turns = [Turn(question="q", answer="a")]
    store = SimpleNamespace(load=lambda sid, limit: turns)
    router = Router(store)
    assert router.load_history("s1", limit=10) == turns


def test_router_load_history_passes_limit() -> None:
    """``Router.load_history(session_id, limit=N)`` forwards N to the store."""

    captured: dict[str, object] = {}

    def fake_load(sid: str, limit: int) -> list[Turn]:
        captured["sid"] = sid
        captured["limit"] = limit
        return []

    router = Router(SimpleNamespace(load=fake_load))
    router.load_history("s1", limit=50)
    assert captured == {"sid": "s1", "limit": 50}


def test_router_record_turn_returns_false_when_session_id_none() -> None:
    """``Router.record_turn`` is a no-op when session_id is None."""

    store = MagicMock()
    router = Router(store)
    turn = SimpleNamespace(question="q", answer="a")
    assert router.record_turn(None, turn) is False
    store.append.assert_not_called()


def test_router_record_turn_skips_empty_answer_when_skip_when_empty_true() -> None:
    """``Router.record_turn(skip_when_empty=True)`` drops empty answers."""

    store = MagicMock()
    router = Router(store)
    turn = SimpleNamespace(question="q", answer="")
    assert router.record_turn("s1", turn) is False
    store.append.assert_not_called()


def test_router_record_turn_persists_non_empty_answer() -> None:
    """``Router.record_turn(skip_when_empty=True)`` persists non-empty answers."""

    captured: dict[str, object] = {}

    class _Store:
        def append(self, sid: str, turn: Any) -> None:
            captured["sid"] = sid
            captured["turn"] = turn

    router = Router(_Store())
    turn = SimpleNamespace(question="q", answer="a")
    assert router.record_turn("s1", turn) is True
    assert captured["sid"] == "s1"
    assert captured["turn"] is turn


def test_router_record_turn_persists_empty_answer_when_skip_when_empty_false() -> None:
    """``Router.record_turn(skip_when_empty=False)`` persists even empty answers."""

    captured: dict[str, object] = {}

    class _Store:
        def append(self, sid: str, turn: Any) -> None:
            captured["sid"] = sid
            captured["turn"] = turn

    router = Router(_Store())
    turn = SimpleNamespace(question="q", answer="")
    assert router.record_turn("s1", turn, skip_when_empty=False) is True
    assert captured["sid"] == "s1"