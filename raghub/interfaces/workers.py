"""Background worker and task queue contracts."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class BackgroundWorker(Protocol):
    """Schedules background tasks."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit a task."""


class TaskQueue(Protocol):
    """Abstract task queue."""

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue a task."""

