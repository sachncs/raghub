"""Background worker and task queue contracts.

Protocols describing the surface area of the project's background
execution layer. The production implementation is
:class:`raghub.services.workers.IngestionWorker`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class BackgroundWorker(Protocol):
    """Schedules background tasks."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit a callable to be executed in the background.

        Args:
            fn: The callable to invoke.
            *args: Positional arguments passed to ``fn``.
            **kwargs: Keyword arguments passed to ``fn``.

        Returns:
            A handle appropriate to the implementation (e.g. a
            ``concurrent.futures.Future``).
        """


class TaskQueue(Protocol):
    """Abstract task queue (e.g. Celery, RQ, SQS)."""

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue a task.

        Args:
            name: The task name.
            payload: Task payload.

        Returns:
            A queue-assigned task id.
        """
