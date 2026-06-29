"""Observability contracts.

Protocols that describe the surface area of the project's logger,
metrics recorder, and tracing/telemetry layer. Concrete
implementations are :class:`raghub.observability.logging.StructuredLogger`,
:class:`raghub.observability.metrics.PrometheusMetrics`, and
:class:`raghub.telemetry.langfuse.LangfuseTelemetryProvider`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable


class Logger(Protocol):
    """Structured logger contract."""

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info-level message."""

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning-level message."""

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error-level message."""


class Metrics(Protocol):
    """Metrics recorder contract."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency metric."""

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment a counter."""


@runtime_checkable
class Span(Protocol):
    """A single open trace span.

    A span is opened via :meth:`TelemetryProvider.start_span` and
    closed via :meth:`end_span`. Spans may be nested.
    """

    name: str

    def end(self) -> None:
        """Close the span."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key/value attribute to the span."""


class TelemetryProvider(Logger, Metrics, Protocol):
    """Combined observability surface: logging + metrics + spans + tokens."""

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a new span; close with :meth:`end_span`."""

    def end_span(self, span: Span) -> None:
        """Close a previously-opened span."""

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage for a model call."""

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context manager wrapper around :meth:`start_span` / :meth:`end_span`."""
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)
