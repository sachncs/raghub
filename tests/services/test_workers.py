"""Tests for ``raghub.services.workers`` (Synchronous, ThreadPool, MemoryQueue)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from raghub.services.workers import MemoryQueue, QueueBase, Synchronous, ThreadPool, Worker


def test_synchronous_submit_invokes_inline() -> None:
    """``Synchronous.submit`` runs the function inline and returns its result."""

    def add(a: int, b: int) -> int:
        return a + b

    assert Synchronous.submit(add, 2, 3) == 5


def test_synchronous_submit_passes_kwargs() -> None:
    """``Synchronous.submit`` forwards kwargs to the wrapped function."""

    def greet(name: str, *, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}"

    assert Synchronous.submit(greet, "alice", greeting="Hi") == "Hi, alice"


def test_worker_default_submit_raises() -> None:
    """``Worker.submit`` raises ``NotImplementedError`` on the abstract base."""

    import inspect

    fn = Worker.__dict__["submit"]
    # fn is the unbound function; calling it bypasses the override.
    with pytest.raises(NotImplementedError):
        fn(None, lambda: None)


def test_queue_base_default_enqueue_raises() -> None:
    """The abstract base QueueBase.enqueue raises NotImplementedError."""

    base = QueueBase()  # type: ignore[abstract]
    with pytest.raises(NotImplementedError):
        base.enqueue("noop", {"any": "payload"})


def test_synchronous_submit_propagates_exception() -> None:
    """``Synchronous.submit`` does not swallow exceptions raised by the function."""

    def boom() -> None:
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError, match="intentional"):
        Synchronous.submit(boom)


def test_thread_pool_submit_returns_future() -> None:
    """``ThreadPool.submit`` returns a concurrent.futures.Future."""

    pool = ThreadPool(max_workers=2)
    try:
        future = pool.submit(lambda x: x * 2, 21)
        assert future.result(timeout=2) == 42
    finally:
        pool.executor.shutdown(wait=True)


def test_thread_pool_runs_concurrently() -> None:
    """Multiple ThreadPool.submit calls execute in parallel."""

    pool = ThreadPool(max_workers=4)
    try:

        def sleep_then_double(x: int) -> int:
            time.sleep(0.05)
            return x * 2

        start = time.perf_counter()
        futures = [pool.submit(sleep_then_double, i) for i in range(4)]
        for f in futures:
            assert f.result(timeout=2) is not None
        elapsed = time.perf_counter() - start
        # Four 50ms sleeps in parallel should take ~50ms total, not 200ms.
        assert elapsed < 0.15
    finally:
        pool.executor.shutdown(wait=True)


def test_memory_queue_enqueue_returns_name() -> None:
    """``MemoryQueue.enqueue`` returns the supplied task name."""

    queue = MemoryQueue()
    name = queue.enqueue("ingest", {"doc": "x"})
    assert name == "ingest"


def test_memory_queue_persists_payload_for_subsequent_get() -> None:
    """Items enqueued can be retrieved in FIFO order via get()."""

    queue = MemoryQueue()
    queue.enqueue("first", {"i": 1})
    queue.enqueue("second", {"i": 2})

    first_name, first_payload = queue.queue.get()
    second_name, second_payload = queue.queue.get()

    assert (first_name, first_payload) == ("first", {"i": 1})
    assert (second_name, second_payload) == ("second", {"i": 2})


def test_memory_queue_is_process_local() -> None:
    """Two MemoryQueue instances have independent stores."""

    queue_a = MemoryQueue()
    queue_b = MemoryQueue()
    queue_a.enqueue("a", {"x": 1})
    queue_b.enqueue("b", {"y": 2})
    assert queue_a.queue.qsize() == 1
    assert queue_b.queue.qsize() == 1
    assert queue_a.queue.get() == ("a", {"x": 1})
    assert queue_b.queue.get() == ("b", {"y": 2})


def test_memory_queue_accepts_arbitrary_payloads() -> None:
    """``MemoryQueue`` accepts any dict-shaped payload, including nested."""

    queue = MemoryQueue()
    payload: dict[str, Any] = {
        "user": "alice",
        "documents": [{"id": "doc-1"}, {"id": "doc-2"}],
        "metadata": {"version": 1, "tags": ["a", "b"]},
    }
    queue.enqueue("batch", payload)
    assert queue.queue.get() == ("batch", payload)
