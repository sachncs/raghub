"""Observability contracts.

Protocols that describe the surface area of the project's logger and
metrics recorder. Concrete implementations are :class:`raghub.observability.logging.StructuredLogger`
and :class:`raghub.observability.metrics.PrometheusMetrics`.
"""

from __future__ import annotations

from typing import Any, Protocol


class Logger(Protocol):
    """Structured logger contract."""

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info-level message.

        Args:
            message: The log message.
            **kwargs: Structured key/value pairs.
        """

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning-level message.

        Args:
            message: The log message.
            **kwargs: Structured key/value pairs.
        """

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error-level message.

        Args:
            message: The log message.
            **kwargs: Structured key/value pairs.
        """


class Metrics(Protocol):
    """Metrics recorder contract."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency metric.

        Args:
            name: Metric name.
            value_ms: Latency in milliseconds.
            **labels: Optional label set.
        """

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment a counter.

        Args:
            name: Counter name.
            value: Increment amount (defaults to 1).
            **labels: Optional label set.
        """
