"""Utility helpers.

This package ships small, dependency-free helpers used across the
codebase:

* :func:`atomic_write_json` — atomic disk write via a temporary file
  and ``os.replace``.
* :func:`load_json` — JSON loader with a sensible default.
* :func:`capture` — exception-aware callable execution.
* :func:`retry` — exponential-backoff retry for transient upstream
  errors.
* :func:`maybe_await` — bridges sync / async callables at API
  boundaries.
* :class:`DurationTimer` — wall-clock timer used by orchestration
  pipelines.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeVar

T = TypeVar("T")


def write_json(payload: Any) -> None:
    """Write ``payload`` as pretty JSON to stdout.

    Args:
        payload: Any JSON-serialisable value.
    """
    sys.stdout.write(json.dumps(payload, indent=2, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON to ``path``.

    The write goes through a sibling temp file followed by
    :func:`os.replace`, which is atomic on POSIX and Windows. This
    prevents readers from observing a partially-written file even
    when the process is killed mid-write.

    Args:
        path: Destination path. Parent directories are created
            automatically.
        payload: The dict to serialize. ``default=str`` is passed to
            :func:`json.dump` so non-JSON-native values fall back to
            their string representation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temp_name = handle.name
    os.replace(temp_name, path)


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load JSON from ``path``, returning ``default`` when missing.

    Args:
        path: Path to the JSON file.
        default: Value returned when ``path`` does not exist. When
            ``None``, an empty dict is returned.

    Returns:
        The parsed dict, or ``default`` when the file is missing.
        Parse errors propagate as :class:`json.JSONDecodeError`.
    """
    if not path.exists():
        return {} if default is None else default
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("JSON root must be an object")
    return decoded


def capture(
    call: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, Exception | None]:
    """Return a callable result and any raised exception."""
    try:
        return call(*args, **kwargs), None
    except Exception as error:
        return None, error


def retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_keywords: tuple[str, ...] = (
        "timeout",
        "rate limit",
        "429",
        "503",
        "500",
        "server error",
        "try again",
    ),
) -> T:
    """Run ``fn`` with exponential-backoff retries on transient errors.

    The function is invoked up to ``max_retries + 1`` times. On each
    failure the exception message is lower-cased and checked for any
    substring in ``retryable_keywords``. If a keyword matches **and** more
    retries are available, the function sleeps for ``base_delay * 2 **
    attempt`` seconds and re-invokes ``fn``. Otherwise the exception
    propagates unchanged.

    Args:
        fn: A zero-argument callable producing the desired return value.
        max_retries: Maximum number of retries **after** the first
            attempt. Total invocations = ``max_retries + 1``.
        base_delay: Initial sleep in seconds. Doubles each attempt; no cap.
        retryable_keywords: Lower-cased substrings that mark an error as
            transient. Default covers HTTP 429/500/503, "rate limit",
            "timeout", and a few common upstream phrasings.

    Returns:
        Whatever ``fn`` returns on a successful attempt.

    Raises:
        Exception: The most recent exception from ``fn``, re-raised once
            the retry budget is exhausted or the error is deemed
            non-retryable.

    Note:
        Sleeping blocks the calling thread. Use the async variant of your
        provider (or wrap in ``asyncio.to_thread``) when calling from a
        coroutine to avoid blocking the event loop.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (ConnectionError, TimeoutError, OSError) as exc:
            msg = str(exc).lower()
            if attempt < max_retries and any(k in msg for k in retryable_keywords):
                time.sleep(base_delay * (2**attempt))
            else:
                raise
    raise RuntimeError("unreachable")


async def aretry(
    fn: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_keywords: tuple[str, ...] = (
        "timeout",
        "rate limit",
        "429",
        "503",
        "500",
        "server error",
        "try again",
    ),
) -> T:
    """Async version of :func:`retry` for use inside coroutines.

    Uses :func:`asyncio.sleep` instead of :func:`time.sleep` so the
    event loop stays responsive during back-off waits.

    Args:
        fn: A zero-argument callable returning an awaitable.
        max_retries: Maximum number of retries **after** the first attempt.
        base_delay: Initial sleep in seconds. Doubles each attempt; no cap.
        retryable_keywords: Lower-cased substrings that mark an error as transient.

    Returns:
        Whatever ``fn()`` returns on a successful attempt.

    Raises:
        Exception: The most recent exception from ``fn``, re-raised once
            the retry budget is exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except (ConnectionError, TimeoutError, OSError) as exc:
            msg = str(exc).lower()
            if attempt < max_retries and any(k in msg for k in retryable_keywords):
                await asyncio.sleep(base_delay * (2**attempt))
            else:
                raise
    raise RuntimeError("unreachable")


async def maybe_await(value: T | Awaitable[T]) -> T:
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


def maybe_await_sync(awaitable: Any) -> Any:
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