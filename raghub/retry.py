"""Sync and async retry helpers with exponential back-off."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

__all__ = ["aretry", "retry"]


def retry[T](
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


async def aretry[T](
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
