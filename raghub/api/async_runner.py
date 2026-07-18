"""Helpers for running async code from sync call-sites.

Provides a single :func:`maybe_await` helper that detects whether a
loop is already running and either awaits the coroutine or
schedules it via :func:`asyncio.run`. Keeps the public ``RAG``
class free of asyncio plumbing.
"""

from __future__ import annotations

import asyncio
import collections.abc
from collections.abc import Awaitable
from typing import Any, cast


def maybe_await(awaitable: Awaitable[Any]) -> Any:
    """Run ``awaitable`` whether or not a loop is already running.

    If a loop is running, returns the coroutine so the caller can
    ``await`` it. Otherwise wraps ``awaitable`` in
    :func:`asyncio.run` so the sync facade still works.

    Args:
        awaitable: The coroutine to schedule.

    Returns:
        Either the resolved value (sync path) or the coroutine
        (async-from-async path).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(cast(collections.abc.Coroutine[Any, Any, Any], awaitable))
    return awaitable
