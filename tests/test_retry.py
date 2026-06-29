"""Tests for the exponential-backoff retry helper."""

from __future__ import annotations

import pytest

from raghub.utils.retry import retry


def test_retry_returns_first_success() -> None:
    """``retry`` returns the value of a successful call without retrying."""

    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    assert retry(fn) == "ok"
    assert len(calls) == 1


def test_retry_retries_transient_then_succeeds() -> None:
    """A transient error is retried until success."""

    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("upstream timeout")
        return "ok"

    assert retry(fn, max_retries=3, base_delay=0) == "ok"
    assert len(calls) == 3


def test_retry_reraises_non_transient() -> None:
    """A non-transient error propagates on the first failure."""

    calls: list[int] = []

    def fn() -> None:
        calls.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        retry(fn, max_retries=3, base_delay=0)
    assert len(calls) == 1


def test_retry_exhausts_budget_and_raises() -> None:
    """When all attempts fail with a transient error, the last one is raised."""

    calls: list[int] = []

    def fn() -> None:
        calls.append(1)
        raise RuntimeError("upstream 503 server error")

    with pytest.raises(RuntimeError, match="upstream 503 server error"):
        retry(fn, max_retries=2, base_delay=0)
    assert len(calls) == 3


def test_retry_matches_retryable_keywords_case_insensitively() -> None:
    """Retryable keywords are matched case-insensitively."""

    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("RATE LIMIT")
        return "ok"

    assert retry(fn, max_retries=2, base_delay=0) == "ok"
    assert len(calls) == 2


def test_retry_custom_retryable_keywords() -> None:
    """Custom retryable keywords are honoured."""

    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("please-retry-me")
        return "ok"

    assert (
        retry(fn, max_retries=2, base_delay=0, retryable_keywords=("please-retry-me",))
        == "ok"
    )
    assert len(calls) == 2
