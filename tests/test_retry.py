"""Tests for retry decorator."""

from __future__ import annotations

import time

from raghub.utils import retry


def test_retry_succeeds_after_failure():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("simulated timeout")
        return "ok"

    result = retry(flaky, max_retries=3, base_delay=0.01)
    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_propagates_non_retryable():
    def always_fails():
        raise ValueError("not retryable")

    try:
        retry(always_fails, max_retries=3, base_delay=0.01)
    except ValueError as exc:
        assert "not retryable" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_retry_respects_keyword_match():
    """A retryable keyword (rate limit) triggers full retry cycle then re-raises."""
    attempts = {"count": 0}

    def keyword_fail():
        attempts["count"] += 1
        raise ConnectionError("rate limit exceeded")

    try:
        retry(keyword_fail, max_retries=2, base_delay=0.01)
    except ConnectionError:
        pass  # Expected after retries are exhausted
    else:
        raise AssertionError("expected ConnectionError after retries")

    # max_retries=2 → 3 total attempts (initial + 2 retries)
    assert attempts["count"] == 3


def test_retry_returns_immediately_on_success():
    attempts = {"count": 0}

    def always_works():
        attempts["count"] += 1
        return 42

    result = retry(always_works, max_retries=3)
    assert result == 42
    assert attempts["count"] == 1