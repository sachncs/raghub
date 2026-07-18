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
"""

from __future__ import annotations  # isort: skip

import time
from typing import Any

from .application import DynamicRagApplication, DynamicRagContainer, build_container
from .workers import InMemoryTaskQueue, SynchronousWorker, ThreadPoolWorker


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
            try:
                log_method(message, extra=payload)
            except TypeError:
                # Some logger implementations don't accept ``extra``;
                # degrade to a plain formatted message so we never lose
                # the event entirely.
                log_method(f"{message} {payload}")

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
            # ``perf_counter`` returns fractional seconds; we want ms to
            # match the histogram buckets configured in metrics.
            recorder(name, (time.perf_counter() - started_at) * 1000.0)


__all__ = [
    "DynamicRagApplication",
    "DynamicRagContainer",
    "InMemoryTaskQueue",
    "ServiceMixin",
    "SynchronousWorker",
    "ThreadPoolWorker",
    "build_container",
]
