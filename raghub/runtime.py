"""Runtime utilities: sync/async boundary helpers and a wall-clock timer.

Two small, single-purpose helper modules are merged here:

- :func:`capture` — return ``(result, exception_or_none)`` from a
  callable, useful for bridging exceptions across sync/async.
- :func:`await_if_awaitable` — bridge sync and async callables at
  API boundaries.
- :func:`run_synchronously` — run a coroutine in a fresh event loop
  when no loop is already running, otherwise surface the coroutine
  for the caller to await.
- :class:`DurationTimer` — record a start instant; ``elapsed_ms``
  returns milliseconds since start.

This module is a leaf of the import graph (no other raghub module
imports it transitively into a cycle). Multiple other modules depend
on these primitives, so it is kept intentionally tiny.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

__all__ = [
    "DurationTimer",
    "await_if_awaitable",
    "capture",
    "run_synchronously",
]


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


async def await_if_awaitable[T](value: T | Awaitable[T]) -> T:
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


def run_synchronously(awaitable: Any) -> Any:
    """Run ``awaitable`` whether or not a loop is already running.

    Sync counterpart to :func:`await_if_awaitable`. If a loop is running,
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


class DurationTimer:
    """Wall-clock timer used by orchestration pipelines.

    Records the start instant on construction; :meth:`elapsed_ms`
    returns milliseconds since the start. Used by ingest/query
    pipelines to publish latency metrics without depending on
    third-party timing libraries.

    The timer is **not** thread-safe; pipelines that fan out to
    threads should construct a fresh :class:`DurationTimer` per
    coroutine.
    """

    def __init__(self) -> None:
        """Record the current ``time.perf_counter`` as the start."""
        self.start = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Return the elapsed time in milliseconds since construction.

        Returns:
            Elapsed time in milliseconds (float).

        """
        return (time.perf_counter() - self.start) * 1000.0
