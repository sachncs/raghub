"""Background worker implementations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Any, Callable

from dynamic_rag.interfaces.workers import BackgroundWorker, TaskQueue


class SynchronousWorker(BackgroundWorker):
    """Execute tasks inline."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)


class ThreadPoolWorker(BackgroundWorker):
    """Execute tasks in a thread pool."""

    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        return self.executor.submit(fn, *args, **kwargs)


class InMemoryTaskQueue(TaskQueue):
    """Simple queue abstraction for future integration with Celery or RQ."""

    def __init__(self) -> None:
        self.queue: Queue[tuple[str, dict[str, Any]]] = Queue()

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        self.queue.put((name, payload))
        return name

