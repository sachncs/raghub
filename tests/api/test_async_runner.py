"""Tests for ``raghub.api.async_runner.maybe_await``."""
from __future__ import annotations

import asyncio

import pytest

from raghub.api.async_runner import maybe_await


# ---------------------------------------------------------------------------
# Sync path: no running loop
# ---------------------------------------------------------------------------


def test_maybe_await_runs_coroutine_in_sync_context() -> None:
    """Without a running loop, the coroutine is executed to completion."""
    result = maybe_await(_coro(7))
    assert result == 7


def test_maybe_await_returns_value_for_sync_caller() -> None:
    """The sync path returns the resolved value (not the coroutine)."""
    out = maybe_await(_coro(0))
    assert out == 0


def test_maybe_await_handles_empty_return() -> None:
    """A coroutine returning ``None`` yields ``None`` in sync mode."""
    assert maybe_await(_empty_coro()) is None


def test_maybe_await_propagates_exceptions() -> None:
    """Exceptions raised by the coroutine propagate to the caller."""
    with pytest.raises(RuntimeError):
        maybe_await(_failing_coro())


def test_maybe_await_supports_complex_return_value() -> None:
    """Dictionaries, lists, and dataclasses round-trip."""
    expected = {"k": [1, 2, 3], "nested": {"x": True}}
    assert maybe_await(_coro(expected)) == expected


# ---------------------------------------------------------------------------
# Async path: a loop is already running
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_await_returns_coroutine_in_running_loop() -> None:
    """With a loop running, ``maybe_await`` returns the coroutine for awaiting."""

    async def _build() -> str:
        return "ready"

    coroutine = _build()
    returned = maybe_await(coroutine)
    assert asyncio.iscoroutine(returned)
    assert await returned == "ready"


@pytest.mark.asyncio
async def test_maybe_await_async_path_returns_same_instance() -> None:
    """In the async path the wrapper passes the coroutine through unchanged."""

    async def _build() -> int:
        return 99

    coroutine = _build()
    try:
        assert maybe_await(coroutine) is coroutine
    finally:
        await coroutine


@pytest.mark.asyncio
async def test_maybe_await_async_path_evaluates_correctly() -> None:
    """Awaiting the returned coroutine yields the expected value."""
    coroutine = _coro("hello")
    assert await maybe_await(coroutine) == "hello"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _coro(value: object):
    return value


async def _empty_coro() -> None:
    return None


async def _failing_coro() -> None:
    raise RuntimeError("kaboom")