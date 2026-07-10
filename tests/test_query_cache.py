"""Unit tests for the in-memory :class:`QueryCache`."""

from __future__ import annotations

import time

import pytest

from raghub.models import PipelineResult
from raghub.pipelines.cache import QueryCache


def _make_result(answer: str = "hello") -> PipelineResult:
    return PipelineResult(
        pipeline_id="p",
        pipeline_name="query",
        success=True,
        outputs={"answer": answer},
    )


def test_set_and_get_returns_same_result() -> None:
    cache = QueryCache(ttl_seconds=60)
    result = _make_result()
    cache.set("q", "user1", {"company": ["Apple"]}, result)
    assert cache.get("q", "user1", {"company": ["Apple"]}) is result


def test_get_returns_none_for_missing_question() -> None:
    cache = QueryCache(ttl_seconds=60)
    assert cache.get("missing", "user1") is None


def test_get_returns_none_for_different_user() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("q", "alice", None, _make_result())
    assert cache.get("q", "bob", None) is None


def test_get_returns_none_for_different_filters() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("q", "alice", {"company": ["Apple"]}, _make_result())
    assert cache.get("q", "alice", {"company": ["Microsoft"]}) is None


def test_filters_are_canonicalised() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("q", "alice", {"company": ["Apple"], "tag": "x"}, _make_result())
    assert cache.get("q", "alice", {"tag": "x", "company": ["Apple"]}) is not None


def test_get_evicts_expired_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = QueryCache(ttl_seconds=10)
    cache.set("q", "alice", None, _make_result())
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base + 11)
    assert cache.get("q", "alice", None) is None
    assert "q" not in cache._store  # type: ignore[attr-defined]


def test_clear_evicts_everything() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("a", "u", None, _make_result())
    cache.set("b", "u", None, _make_result())
    cache.clear()
    assert cache.get("a", "u", None) is None
    assert cache.get("b", "u", None) is None


def test_invalidate_full_clears_cache() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("a", "u1", None, _make_result())
    cache.set("b", "u2", None, _make_result())
    cache.invalidate()
    assert cache.get("a", "u1", None) is None
    assert cache.get("b", "u2", None) is None


def test_invalidate_by_question() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("a", "u1", None, _make_result("alpha"))
    cache.set("b", "u1", None, _make_result("beta"))
    cache.invalidate(question="a")
    assert cache.get("a", "u1", None) is None
    assert cache.get("b", "u1", None) is not None


def test_invalidate_by_user() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("a", "u1", None, _make_result("alpha"))
    cache.set("a", "u2", None, _make_result("beta"))
    cache.invalidate(user_id="u1")
    assert cache.get("a", "u1", None) is None
    assert cache.get("a", "u2", None) is not None


def test_invalidate_by_question_and_user() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("a", "u1", None, _make_result("alpha"))
    cache.set("a", "u2", None, _make_result("beta"))
    cache.invalidate(question="a", user_id="u1")
    assert cache.get("a", "u1", None) is None
    assert cache.get("a", "u2", None) is not None


def test_get_with_none_user_id() -> None:
    cache = QueryCache(ttl_seconds=60)
    cache.set("q", None, None, _make_result())
    assert cache.get("q", None, None) is not None