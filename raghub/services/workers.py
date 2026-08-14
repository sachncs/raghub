"""Worker primitives: synchronous, thread-pool, and memory-queue shims.

Every worker/queue registers itself with :class:`Worker` (or
:class:`Queue` for queue-shaped executors) under a stable name.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Any

from raghub.registry import Registry
from raghub.types import JSONValue


class Worker(Registry):
    """Polymorphic base for in-process task executors.

    Concrete workers register themselves via ``@Worker.register`` and
    implement :meth:`submit`.
    """

    def submit(
        self,
        fn: Callable[..., JSONValue],
        *args: JSONValue,
        **kwargs: JSONValue,
    ) -> Any:
        """Submit ``fn`` for execution; return a handle."""
        raise NotImplementedError


class QueueBase(Registry):
    """Polymorphic base for persistent or in-memory task queues.

    Concrete queues register themselves via ``@QueueBase.register`` and
    implement :meth:`enqueue`.
    """

    name: str = "queue"

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue ``payload`` under ``name``; return its job id."""
        raise NotImplementedError


@Worker.register("synchronous")
class Synchronous(Worker):
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


@Worker.register("threadpool")
class ThreadPool(Worker):
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


@QueueBase.register("memory")
class MemoryQueue(QueueBase):
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


__all__ = ["MemoryQueue", "QueueBase", "Synchronous", "ThreadPool", "Worker"]
