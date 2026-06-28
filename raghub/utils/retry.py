"""Retry with exponential backoff for transient failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_keywords: tuple[str, ...] = ("timeout", "rate limit", "429", "503", "500", "server error", "try again"),
) -> T:
    """Retry a callable with exponential backoff on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if attempt < max_retries and any(k in msg for k in retryable_keywords):
                time.sleep(base_delay * (2**attempt))
            else:
                raise
    raise last_exc  # type: ignore[misc]
