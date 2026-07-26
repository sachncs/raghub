"""Application services and shared mixin.

This package exposes the high-level service classes
(:class:`AuthService`, :class:`DocumentService`, :class:`QueryService`,
:class:`HealthService`), the :class:`DynamicRagApplication` facade, and
the in-process worker primitives used by background jobs.

The :class:`ServiceMixin` defined here is mixed into every service so they
all get a uniform ``log()`` and ``emit_metric()`` API. The mixin tolerates
collaborators that don't implement the optional ``info`` /
``record_latency`` methods, which keeps services usable in tests where
the container holds lightweight stubs.

Importing the package alone does **not** eagerly import the
application facade or the workers — those are exposed via
``__getattr__`` so a service module that imports :class:`ServiceMixin`
does not pull in the application package (and the cycle it would
trigger).
"""

from __future__ import annotations

import time
from typing import Any


class ServiceMixin:
    """Provides structured logging and metric helpers to service classes.

    Both methods gracefully degrade when the container is missing the
    expected collaborators, so services can be exercised with stub
    containers in tests.

    Attributes:
        container: The :class:`DynamicRagContainer` (or compatible stub)
            providing ``logger`` and ``metrics`` attributes.
    """

    container: Any

    def log(self, message: str, **payload: Any) -> None:
        """Emit a structured log event.

        Looks up ``container.logger.info`` and calls it with
        ``extra=payload``. If the logger rejects the ``extra`` argument
        (some loggers don't accept it) we fall back to embedding the
        payload into the message string.

        Args:
            message: The event name or short description.
            **payload: Structured fields attached to the log event.
        """
        logger = getattr(self.container, "logger", None)
        log_method = getattr(logger, "info", None) if logger else None
        if callable(log_method):
            log_method(message, extra=payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Record a latency metric given a ``perf_counter`` start time.

        Args:
            name: Metric name (e.g. ``"query_latency_ms"``).
            started_at: Value previously returned by
                :func:`time.perf_counter` at the start of the operation.
        """
        metrics = getattr(self.container, "metrics", None)
        recorder = getattr(metrics, "record_latency", None) if metrics else None
        if callable(recorder):
            recorder(name, (time.perf_counter() - started_at) * 1000.0)


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DynamicRagApplication": ("raghub.services.application", "DynamicRagApplication"),
    "DynamicRagContainer": ("raghub.services.application", "DynamicRagContainer"),
    "build_container": ("raghub.services.application", "build_container"),
    "InMemoryTaskQueue": ("raghub.services.workers", "InMemoryTaskQueue"),
    "SynchronousWorker": ("raghub.services.workers", "SynchronousWorker"),
    "ThreadPoolWorker": ("raghub.services.workers", "ThreadPoolWorker"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose the application facade and worker primitives.

    Args:
        name: One of ``DynamicRagApplication``, ``DynamicRagContainer``,
            ``build_container``, ``InMemoryTaskQueue``,
            ``SynchronousWorker``, or ``ThreadPoolWorker``.

    Returns:
        The corresponding object from the :mod:`raghub.services.application`
        or :mod:`raghub.services.workers` submodule.

    Raises:
        AttributeError: When ``name`` is not a known lazy attribute.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'raghub.services' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


__all__ = [
    "DynamicRagApplication",
    "DynamicRagContainer",
    "InMemoryTaskQueue",
    "ServiceMixin",
    "SynchronousWorker",
    "ThreadPoolWorker",
    "build_container",
]