"""Sync/async boundary helpers and exception-aware callable execution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

__all__ = ["capture", "maybe_await", "maybe_run"]


def capture(
    call: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, Exception | None]:
    """Return a callable result and any raised exception.

    The callable's return type is intentionally ``Any`` rather than
    :data:`JSONValue`: legitimate callers pass functions whose return
    type is ``Settings``, ``PackageMetadata``, ``Module``, or another
    non-JSON-serialisable object. The capture itself does not
    serialise; it only shields the caller from exceptions.
    """
    try:
        return call(*args, **kwargs), None
    except Exception as error:
        return None, error


async def maybe_await[T](value: T | Awaitable[T]) -> T:
    """Await ``value`` if it is awaitable; otherwise return it as-is.

    Bridges sync and async callables at API boundaries so a single
    function can accept either flavour. Coroutines, ``asyncio.Future``
    instances, and any object exposing ``__await__`` are awaited.

    Args:
        value: Either a concrete result or an awaitable that resolves
            to the result.

    Returns:
        The awaited-or-direct result.

    """
    if inspect.isawaitable(value):
        return await value
    return value


def maybe_run(awaitable: Any) -> Any:
    """Run ``awaitable`` whether or not a loop is already running.

    Sync counterpart to :func:`maybe_await`. If a loop is running,
    returns the coroutine so the caller can ``await`` it. Otherwise
    wraps ``awaitable`` in :func:`asyncio.run` so the sync facade
    still works.

    Args:
        awaitable: The coroutine to schedule.

    Returns:
        Either the resolved value (sync path) or the coroutine
        (async-from-async path).

    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    return awaitable
