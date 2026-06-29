"""Structlog + Prometheus adapter for :class:`TelemetryProvider`."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from raghub.interfaces.observability import Span, TelemetryProvider
from raghub.observability.logging import StructuredLogger
from raghub.observability.metrics import PrometheusMetrics
from raghub.telemetry.langfuse import NoopSpan


class StructlogSpan(Span):
    """Span whose :meth:`end` and :meth:`set_attribute` record locally.

    Used by :class:`StructlogTelemetryProvider` so that spans are
    observable through structlog + Prometheus even when no remote
    tracing backend is configured.
    """

    def __init__(self, name: str, logger: StructuredLogger, metrics: PrometheusMetrics) -> None:
        self.name = name
        self._logger = logger
        self._metrics = metrics
        self._attrs: dict[str, Any] = {}
        self._started = time.perf_counter()

    def end(self) -> None:
        """Record the span duration and log its completion."""
        duration_ms = (time.perf_counter() - self._started) * 1000.0
        self._metrics.record_latency(f"span.{self.name}", duration_ms, **self._attrs)
        self._logger.info(f"span.end.{self.name}", duration_ms=duration_ms, **self._attrs)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute for later emission."""
        self._attrs[key] = value


class StructlogTelemetryProvider(TelemetryProvider):
    """Adapter that satisfies the new contract via the legacy stack."""

    def __init__(
        self,
        *,
        logger: StructuredLogger | None = None,
        metrics: PrometheusMetrics | None = None,
    ) -> None:
        """Initialise the provider with a logger and metrics sink.

        Args:
            logger: Optional pre-built :class:`StructuredLogger`.
            metrics: Optional pre-built :class:`PrometheusMetrics`.
        """
        from raghub.observability.logging import build_logger
        from raghub.observability.metrics import PrometheusMetrics

        self._logger = logger or StructuredLogger(build_logger("INFO"))
        self._metrics = metrics or PrometheusMetrics()

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an info log.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.
        """
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Emit a warning log.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.
        """
        self._logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an error log.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.
        """
        self._logger.error(message, **kwargs)

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency metric.

        Args:
            name: Metric name.
            value_ms: Latency in milliseconds.
            **labels: Optional label set.
        """
        self._metrics.record_latency(name, value_ms, **labels)

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment a counter.

        Args:
            name: Counter name.
            value: Increment amount.
            **labels: Optional label set.
        """
        self._metrics.increment(name, value, **labels)

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a span.

        Args:
            name: Span name.
            **attrs: Span attributes.

        Returns:
            A :class:`StructlogSpan` (or :class:`NoopSpan` on error).
        """
        try:
            span = StructlogSpan(name, self._logger, self._metrics)
            for key, value in attrs.items():
                span.set_attribute(key, value)
            return span
        except Exception:
            return NoopSpan(name)

    def end_span(self, span: Span) -> None:
        """Close a span.

        Args:
            span: The span returned by :meth:`start_span`.
        """
        if isinstance(span, NoopSpan):
            return
        span.end()

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage.

        Args:
            name: Generation name.
            prompt_tokens: Input token count.
            completion_tokens: Output token count.
            model: Model identifier.
        """
        self._metrics.increment("tokens.prompt", prompt_tokens, model=model)
        self._metrics.increment("tokens.completion", completion_tokens, model=model)
        self._logger.info(
            "tokens",
            name=name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context-manager wrapper around :meth:`start_span` / :meth:`end_span`.

        Args:
            name: Span name.
            **attrs: Span attributes.

        Yields:
            The open :class:`Span`.
        """
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)


__all__ = ["StructlogTelemetryProvider", "StructlogSpan"]
