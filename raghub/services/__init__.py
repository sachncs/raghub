"""Application services."""

from __future__ import annotations  # isort: skip

from time import perf_counter
from typing import Any


class ServiceMixin:
    """Mixin providing log() and emit_metric() for service classes."""

    container: Any

    def log(self, message: str, **payload: Any) -> None:
        logger = getattr(self.container, "logger", None)
        log_method = getattr(logger, "info", None) if logger else None
        if callable(log_method):
            try:
                log_method(message, extra=payload)
            except TypeError:
                log_method(f"{message} {payload}")

    def emit_metric(self, name: str, started_at: float) -> None:
        metrics = getattr(self.container, "metrics", None)
        recorder = getattr(metrics, "record_latency", None) if metrics else None
        if callable(recorder):
            recorder(name, (perf_counter() - started_at) * 1000.0)


from .application import DynamicRagApplication, DynamicRagContainer, build_container  # noqa: E402
from .workers import InMemoryTaskQueue, SynchronousWorker, ThreadPoolWorker  # noqa: E402

__all__ = [
    "DynamicRagApplication",
    "DynamicRagContainer",
    "InMemoryTaskQueue",
    "ServiceMixin",
    "SynchronousWorker",
    "ThreadPoolWorker",
    "build_container",
]
