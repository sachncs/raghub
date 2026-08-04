"""Coverage tests for :mod:`raghub.utils` edge cases."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from raghub.await_sync import capture, maybe_await, maybe_run
from raghub.io import atomic_write_json, load_json, write_json
from raghub.retry import aretry, retry
from raghub.timing import DurationTimer

# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


def test_load_json_root_must_be_dict(tmp_path: Path) -> None:
    """A non-object JSON root raises :class:`TypeError`."""
    path = tmp_path / "data.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(TypeError, match="root must be an object"):
        load_json(path)


def test_load_json_default_when_missing(tmp_path: Path) -> None:
    """A missing file returns the default (``{}`` when none supplied)."""
    assert load_json(tmp_path / "missing.json") == {}


def test_load_json_explicit_default_when_missing(tmp_path: Path) -> None:
    """A missing file returns the supplied default."""
    sentinel: dict[str, Any] = {"sentinel": True}
    assert load_json(tmp_path / "missing.json", default=sentinel) is sentinel


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


def test_retry_returns_on_first_success() -> None:
    """A function that never raises is invoked once."""
    calls: list[int] = []

    def _fn() -> str:
        calls.append(1)
        return "ok"

    assert retry(_fn, max_retries=3, base_delay=0.0) == "ok"
    assert len(calls) == 1


def test_retry_eventually_succeeds() -> None:
    """A function that fails twice then succeeds is retried until success."""
    counter: dict[str, int] = {"n": 0}

    def _fn() -> str:
        counter["n"] += 1
        if counter["n"] < 3:
            raise TimeoutError("rate limit exceeded")
        return "finally"

    assert retry(_fn, max_retries=5, base_delay=0.0) == "finally"
    assert counter["n"] == 3


def test_retry_gives_up_after_max_retries() -> None:
    """A function that always raises is retried ``max_retries + 1`` times."""
    counter: dict[str, int] = {"n": 0}

    def _fn() -> str:
        counter["n"] += 1
        raise TimeoutError("server error 500")

    with pytest.raises(TimeoutError, match="server error 500"):
        retry(_fn, max_retries=2, base_delay=0.0)
    assert counter["n"] == 3


def test_retry_non_retryable_raises_immediately() -> None:
    """A non-retryable error type is not retried."""
    counter: dict[str, int] = {"n": 0}

    def _fn() -> str:
        counter["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        retry(_fn, max_retries=5, base_delay=0.0)
    assert counter["n"] == 1


def test_retry_timeout_without_keyword_raises_immediately() -> None:
    """A retryable error type without a keyword is not retried."""
    counter: dict[str, int] = {"n": 0}

    def _fn() -> str:
        counter["n"] += 1
        raise TimeoutError("something else entirely")

    with pytest.raises(TimeoutError, match="something else"):
        retry(_fn, max_retries=5, base_delay=0.0)
    assert counter["n"] == 1


# ---------------------------------------------------------------------------
# aretry
# ---------------------------------------------------------------------------


async def test_aretry_returns_on_first_success() -> None:
    """An async function that never raises is invoked once."""

    async def _fn() -> str:
        return "ok"

    assert await aretry(_fn, max_retries=2, base_delay=0.0) == "ok"


async def test_aretry_eventually_succeeds() -> None:
    """An async function that fails twice then succeeds is retried."""
    counter: dict[str, int] = {"n": 0}

    async def _fn() -> str:
        counter["n"] += 1
        if counter["n"] < 3:
            raise TimeoutError("timeout")
        return "ok"

    assert await aretry(_fn, max_retries=5, base_delay=0.0) == "ok"
    assert counter["n"] == 3


async def test_aretry_gives_up_after_max_retries() -> None:
    """An async function that always raises is retried until exhaustion."""
    counter: dict[str, int] = {"n": 0}

    async def _fn() -> str:
        counter["n"] += 1
        raise TimeoutError("rate limit 429")

    with pytest.raises(TimeoutError, match="rate limit"):
        await aretry(_fn, max_retries=2, base_delay=0.0)
    assert counter["n"] == 3


async def test_aretry_non_retryable_raises_immediately() -> None:
    """A non-retryable error is not retried."""
    counter: dict[str, int] = {"n": 0}

    async def _fn() -> str:
        counter["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await aretry(_fn, max_retries=5, base_delay=0.0)
    assert counter["n"] == 1


# ---------------------------------------------------------------------------
# maybe_await / maybe_run
# ---------------------------------------------------------------------------


async def test_maybe_await_returns_value_unchanged() -> None:
    """A plain value is returned as-is."""
    assert await maybe_await("hello") == "hello"
    assert await maybe_await(42) == 42


async def test_maybe_await_awaits_coroutine() -> None:
    """A coroutine is awaited and its result returned."""

    async def _coro() -> str:
        return "from-coroutine"

    assert await maybe_await(_coro()) == "from-coroutine"


def test_maybe_run_runs_coroutine_when_no_loop() -> None:
    """``maybe_run`` runs the coroutine when no event loop is running."""

    async def _coro() -> str:
        return "ran-via-asyncio-run"

    assert maybe_run(_coro()) == "ran-via-asyncio-run"


def test_maybe_run_returns_coroutine_when_loop_active() -> None:
    """``maybe_run`` returns the coroutine when a loop is already running."""
    import asyncio

    async def _coro() -> str:
        return "ran-elsewhere"

    async def _runner() -> Any:
        coro = _coro()
        result = maybe_run(coro)
        assert asyncio.iscoroutine(result)
        return await result

    assert asyncio.run(_runner()) == "ran-elsewhere"


# ---------------------------------------------------------------------------
# DurationTimer
# ---------------------------------------------------------------------------


def test_duration_timer_records_elapsed() -> None:
    """``DurationTimer.elapsed_ms`` reports the wall-clock duration."""
    timer = DurationTimer()
    time.sleep(0.001)
    elapsed = timer.elapsed_ms()
    assert elapsed > 0
    assert elapsed < 5000


def test_duration_timer_reports_zero_immediately() -> None:
    """``elapsed_ms`` is non-negative even when called immediately."""
    timer = DurationTimer()
    assert timer.elapsed_ms() >= 0


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------


def test_write_json_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """``write_json`` writes pretty JSON to stdout."""
    write_json({"a": 1, "b": [2, 3]})
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed == {"a": 1, "b": [2, 3]}


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


def test_atomic_write_json_creates_file(tmp_path: Path) -> None:
    """``atomic_write_json`` writes the file in the destination path."""
    path = tmp_path / "nested" / "data.json"
    atomic_write_json(path, {"a": 1})
    assert path.exists()
    assert json.loads(path.read_text()) == {"a": 1}


def test_atomic_write_json_overwrites_existing(tmp_path: Path) -> None:
    """``atomic_write_json`` overwrites an existing file."""
    path = tmp_path / "data.json"
    atomic_write_json(path, {"old": True})
    atomic_write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def test_capture_returns_value_and_no_error() -> None:
    """A successful call returns the value and ``None`` error."""
    result, error = capture(lambda: 42)
    assert result == 42
    assert error is None


def test_capture_returns_none_and_error() -> None:
    """A failing call returns ``None`` and the exception."""
    result, error = capture(lambda: 1 / 0)
    assert result is None
    assert isinstance(error, ZeroDivisionError)
