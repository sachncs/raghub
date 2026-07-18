"""Observability contracts.

Protocols that describe the surface area of the project's logger,
metrics recorder, and tracing/telemetry layer. Concrete
implementations are :class:`raghub.observability.logging.LoguruLogger`,
:class:`raghub.observability.metrics.PrometheusMetrics`, and
:class:`raghub.telemetry.langfuse.LangfuseTelemetryProvider`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable


class Logger(Protocol):
    """Structured logger contract."""

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info-level message.

        Args:
            message: The log message.
            **kwargs: Structured key-value pairs attached to the event.
        """

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning-level message.

        Args:
            message: The log message.
            **kwargs: Structured key-value pairs attached to the event.
        """

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error-level message.

        Args:
            message: The log message.
            **kwargs: Structured key-value pairs attached to the event.
        """


class Metrics(Protocol):
    """Metrics recorder contract."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency metric.

        Args:
            name: Metric name (dot-separated, e.g. ``query.duration_ms``).
            value_ms: Observed duration in milliseconds.
            **labels: Dimension labels for the metric.
        """

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment a counter.

        Args:
            name: Metric name (dot-separated, e.g. ``query.total``).
            value: Amount to increment by (default 1).
            **labels: Dimension labels for the metric.
        """


@runtime_checkable
class Span(Protocol):
    """A single open trace span.

    A span is opened via :meth:`TelemetryProvider.start_span` and
    closed via :meth:`end_span`. Spans may be nested.
    """

    name: str

    def end(self) -> None:
        """Close the span.

        Must be called exactly once after :meth:`TelemetryProvider.start_span`.
        Implementations should be idempotent.
        """

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key/value attribute to the span.

        Args:
            key: Attribute name.
            value: Attribute value. Should be JSON-serialisable.
        """


class TelemetryProvider(Logger, Metrics, Protocol):
    """Combined observability surface: logging + metrics + spans + tokens."""

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a new span; close with :meth:`end_span`.

        Args:
            name: Span name (dot-separated, e.g. ``ingest.convert``).
            **attrs: Initial attributes attached to the span.

        Returns:
            The newly-created :class:`Span` instance.
        """

    def end_span(self, span: Span) -> None:
        """Close a previously-opened span.

        Args:
            span: The span returned by :meth:`start_span`.
        """

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage for a model call.

        Args:
            name: Span or metric name (e.g. ``query.generate``).
            prompt_tokens: Number of prompt (input) tokens.
            completion_tokens: Number of completion (output) tokens.
            model: Model identifier (e.g. ``gpt-4``).
        """

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context manager wrapper around :meth:`start_span` / :meth:`end_span`.

        Args:
            name: Span name.
            **attrs: Initial attributes.

        Yields:
            The opened :class:`Span` instance. The span is
            automatically closed when the context exits.
        """
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)
