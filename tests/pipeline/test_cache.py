"""Tests for ``raghub.pipeline.cache``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from raghub.models import Pipeline
from raghub.pipeline.cache import Cache


def test_cache_get_returns_none_on_empty() -> None:
    """``Cache.get`` returns None when nothing is cached."""

    cache = Cache()
    assert cache.get("any question") is None


def test_cache_set_then_get_round_trip() -> None:
    """``Cache.set`` followed by ``Cache.get`` returns the stored pipeline."""

    cache = Cache(ttl_seconds=60)
    pipeline = Pipeline(pipeline_id="abc", pipeline_name="test", metadata={})
    cache.set("revenue", user_id="alice", filters=None, result=pipeline)
    assert cache.get("revenue", user_id="alice") is pipeline


def test_cache_get_evicts_expired_entry() -> None:
    """``Cache.get`` removes entries older than ttl and returns None."""

    cache = Cache(ttl_seconds=60)
    pipeline = Pipeline(pipeline_id="abc", pipeline_name="test", metadata={})
    with patch("time.monotonic", return_value=100.0):
        cache.set("revenue", user_id=None, filters=None, result=pipeline)
    with patch("time.monotonic", return_value=1000.0):
        assert cache.get("revenue") is None
        # The expired entry should be gone from the store.
        assert cache.store == {}


def test_cache_clear_evicts_everything() -> None:
    """``Cache.clear`` empties the store."""

    cache = Cache()
    pipeline = Pipeline(pipeline_id="abc", pipeline_name="test", metadata={})
    cache.set("q1", user_id=None, filters=None, result=pipeline)
    cache.set("q2", user_id=None, filters=None, result=pipeline)
    cache.clear()
    assert cache.store == {}


def test_cache_invalidate_by_question() -> None:
    """``Cache.invalidate(question=...)`` evicts only matching entries."""

    cache = Cache()
    cache.set("q1", user_id=None, filters=None, result=Pipeline(pipeline_id="a", pipeline_name="test", metadata={}))
    cache.set("q2", user_id=None, filters=None, result=Pipeline(pipeline_id="b", pipeline_name="test", metadata={}))
    cache.invalidate(question="q1")
    assert list(cache.store.keys()) == [Cache.make_key("q2", None, None)]


def test_cache_invalidate_by_user_id() -> None:
    """``Cache.invalidate(user_id=...)`` evicts all entries for that user."""

    cache = Cache()
    pipeline_factory = lambda pid: Pipeline(pipeline_id=pid, pipeline_name="test", metadata={})
    cache.set("q1", user_id="alice", filters=None, result=pipeline_factory("a"))
    cache.set("q2", user_id="bob", filters=None, result=pipeline_factory("b"))
    cache.invalidate(user_id="alice")
    keys = list(cache.store.keys())
    assert len(keys) == 1
    assert keys[0][1] == "bob"


def test_cache_invalidate_all_when_no_filters() -> None:
    """``Cache.invalidate()`` with no args is equivalent to ``clear``."""

    cache = Cache()
    cache.set("q1", user_id=None, filters=None, result=Pipeline(pipeline_id="a", pipeline_name="test", metadata={}))
    cache.invalidate()
    assert cache.store == {}


def test_cache_key_distinguishes_users() -> None:
    """``Cache.make_key`` produces different keys for different users."""

    k1 = Cache.make_key("q", user_id="alice", filters=None)
    k2 = Cache.make_key("q", user_id="bob", filters=None)
    assert k1 != k2


def test_cache_key_distinguishes_filters() -> None:
    """``Cache.make_key`` produces different keys for different filters."""

    k1 = Cache.make_key("q", user_id=None, filters={"company": "acme"})
    k2 = Cache.make_key("q", user_id=None, filters={"company": "globex"})
    assert k1 != k2


def test_cache_key_distinguishes_top_k() -> None:
    """``Cache.make_key`` differentiates by top_k option."""

    k1 = Cache.make_key("q", user_id=None, filters=None, top_k=5)
    k2 = Cache.make_key("q", user_id=None, filters=None, top_k=10)
    assert k1 != k2


def test_cache_key_distinguishes_history() -> None:
    """``Cache.make_key`` differentiates by history turns."""

    k1 = Cache.make_key("q", user_id=None, filters=None, history=[SimpleNamespace(question="a", answer="b")])
    k2 = Cache.make_key("q", user_id=None, filters=None, history=[])
    assert k1 != k2