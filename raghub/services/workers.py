"""Worker primitives: synchronous, thread-pool, and memory-queue shims."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Any

from raghub.types import JSONValue


class Synchronous:
    """Execute tasks inline on the caller's thread.

    Useful for tests that want deterministic ordering. Exceptions
    propagate to the caller unchanged.
    """

    @staticmethod
    def submit(fn: Callable[..., JSONValue], *args: JSONValue, **kwargs: JSONValue) -> JSONValue:
        """Invoke ``fn(*args, **kwargs)`` and return its result directly."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            raise


class ThreadPool:
    """Execute tasks on a :class:`ThreadPoolExecutor`.

    Attributes:
        executor: Backing thread pool.

    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialise the worker pool.

        Args:
            max_workers: Maximum concurrent worker threads.

        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        fn: Callable[..., JSONValue],
        *args: JSONValue,
        **kwargs: JSONValue,
    ) -> Future[JSONValue]:
        """Submit ``fn`` to the pool and return its :class:`Future`."""
        return self.executor.submit(fn, *args, **kwargs)


class MemoryQueue:
    """In-memory queue shim intended for Celery/RQ migration.

    Process-local; does not survive restarts.
    """

    def __init__(self) -> None:
        """Initialise the queue."""
        self.queue: Queue[tuple[str, dict[str, Any]]] = Queue()

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue ``payload`` under ``name`` and return ``name``."""
        self.queue.put((name, payload))
        return name


__all__ = ["MemoryQueue", "Synchronous", "ThreadPool"]
