"""Exponential-backoff retry helper for transient failure recovery.

This module provides a single function, :func:`retry`, that runs a
zero-argument callable and re-invokes it with growing sleeps between
attempts if the raised exception's message matches any of a configurable
list of "transient" keywords. It is used by the LLM and embedding
providers to ride out momentary network blips, rate-limit responses, and
upstream 5xx errors without surfacing them to the caller.

Design notes and trade-offs:

* **Substring matching, not exception types.** We deliberately do not
  enumerate HTTP status codes or wrap exceptions in typed hierarchies;
  this keeps :func:`retry` decoupled from ``requests``/``httpx``/SDK
  specifics. The trade-off is a fragile contract: a future error message
  that *doesn't* contain a keyword will be treated as non-retryable even
  if it actually is. If you need typed retries, switch to ``tenacity``.
* **Exponential backoff without jitter or cap.** The sleep grows as
  ``base_delay * 2 ** attempt``. There is no jitter, which means a
  thundering-herd of simultaneous callers can resync their retries. There
  is no cap, so long ``max_retries`` values combined with a generous
  ``base_delay`` will sleep for a long time on the last attempt.
* **Bare ``except Exception`` is intentional.** We catch everything that
  propagates out of ``fn`` because the call site may wrap any provider's
  error type. The keyword check below then short-circuits to a re-raise
  for non-retryable failures.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


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

    Example:
        >>> import random
        >>> def flaky():
        ...     if random.random() < 0.5:
        ...         raise RuntimeError("upstream timeout")
        ...     return 42
        >>> # Don't actually run this in a doctest; it's random.
        >>> # Instead, exercise the deterministic path:
        >>> retry(lambda: "ok")
        'ok'

    Note:
        Sleeping blocks the calling thread. Use the async variant of your
        provider (or wrap in ``asyncio.to_thread``) when calling from a
        coroutine to avoid blocking the event loop.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            # Only sleep+retry if (a) we have budget left and (b) the error
            # message looks transient. The bare ``raise`` on the else branch
            # re-raises the original exception unchanged, preserving the
            # traceback and any chained context.
            if attempt < max_retries and any(k in msg for k in retryable_keywords):
                # Exponential growth: 1x, 2x, 4x, ... of base_delay.
                # No jitter: callers that fan out should add their own
                # randomisation or use a distributed rate limiter.
                time.sleep(base_delay * (2**attempt))
            else:
                raise
    # Unreachable in practice: the loop either returns or re-raises. The
    # ``type: ignore`` acknowledges that mypy cannot prove this.
    raise last_exc  # type: ignore[misc]
