"""In-process worker primitives for fire-and-forget tasks.

This module ships three small abstractions matching the
:class:`raghub.interfaces.workers.BackgroundWorker` and
:class:`TaskQueue` interfaces:

* :class:`SynchronousWorker` — runs the task inline on the caller thread.
  Useful for unit tests and for debugging race conditions without the
  indirection of a thread pool.
* :class:`ThreadPoolWorker` — runs the task on a
  :class:`concurrent.futures.ThreadPoolExecutor`. Returns a
  :class:`concurrent.futures.Future` so callers can compose with the
  broader ``concurrent.futures`` ecosystem.
* :class:`InMemoryTaskQueue` — a ``Queue``-backed shim for code that
  expects a queue abstraction. Intended as the integration point for a
  real broker (Celery, RQ, Dramatiq); the in-memory queue does not
  survive a process restart.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Any

from raghub.interfaces.workers import BackgroundWorker, TaskQueue


class SynchronousWorker(BackgroundWorker):
    """Execute tasks inline on the caller's thread.

    Useful for tests that want deterministic ordering and synchronous
    exception propagation.
    """

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke ``fn(*args, **kwargs)`` and return its result directly.

        Args:
            fn: The callable to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The callable's return value (no ``Future`` wrapping).
        """
        return fn(*args, **kwargs)


class ThreadPoolWorker(BackgroundWorker):
    """Execute tasks on a thread pool.

    Attributes:
        executor: Backing :class:`ThreadPoolExecutor`.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialise the worker.

        Args:
            max_workers: Maximum concurrent worker threads. Default 4.
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit ``fn`` to the pool and return its :class:`Future`.

        Args:
            fn: The callable to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A :class:`concurrent.futures.Future` for the submitted task.
        """
        return self.executor.submit(fn, *args, **kwargs)


class InMemoryTaskQueue(TaskQueue):
    """In-memory queue shim intended for Celery/RQ migration.

    The queue is process-local and does **not** survive restarts. Use it
    only as a development convenience or as the integration target when
    swapping in a real broker.

    Attributes:
        queue: Underlying :class:`queue.Queue` of ``(name, payload)`` tuples.
    """

    def __init__(self) -> None:
        """Initialise the queue."""
        self.queue: Queue[tuple[str, dict[str, Any]]] = Queue()

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue ``payload`` under ``name`` and return ``name``.

        Args:
            name: Task identifier (also used as the returned handle).
            payload: Arbitrary serialisable task arguments.

        Returns:
            The task name, suitable as a job handle for status lookup.
        """
        self.queue.put((name, payload))
        return name
