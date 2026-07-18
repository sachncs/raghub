"""Loguru-backed logger and metrics adapter.

This module replaces the previous structlog adapter with a thin
wrapper around :mod:`loguru`. The public surface is intentionally
small:

* :func:`build_logger` — configure the process-wide loguru sink and
  return the bound :class:`LoguruLogger`.
* :class:`LoguruLogger` — implements the
  :class:`raghub.interfaces.observability.Logger` protocol so
  service modules do not import loguru directly.
* :class:`LoguruTelemetryProvider` — adapts the protocol to the
  loguru pipeline; ``span()`` returns a context manager that records
  span duration into the metrics layer when closed.

The :data:`SECRET_KEY_RE` is shared with the redacting telemetry
wrapper. Loguru is configured with a sink that scrubs secret-looking
keys before the record is emitted, so callers do not have to
hand-roll redaction in every code path.

Example:
    >>> from raghub.observability.logging import build_logger
    >>> logger = build_logger()
    >>> logger.info("startup.complete", port=8000)
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger as loguru_logger

from raghub.interfaces.observability import Logger, Metrics, Span, TelemetryProvider
from raghub.observability.metrics import PrometheusMetrics

SECRET_KEY_RE = re.compile(
    r"(?i)(password|passwd|secret|api_key|apikey|access_token|refresh_token|jwt|authorization)"
)


def redact_record(record: dict[str, Any]) -> None:
    """In-place redact secret-looking values in a loguru record.

    Args:
        record: The mutable loguru ``record.message`` dictionary; values
            whose key matches :data:`SECRET_KEY_RE` are replaced by
            ``"***"``. Nested dicts are scrubbed recursively.
    """

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    scrubbed: dict[str, Any] = {}
    for key, value in record.items():
        if SECRET_KEY_RE.search(str(key)):
            scrubbed[key] = "***"
        else:
            scrubbed[key] = scrub(value)
    record.clear()
    record.update(scrubbed)


def build_logger(level: str = "INFO") -> LoguruLogger:
    """Configure the process-wide loguru logger.

    Removes any default sinks installed by loguru's ``logger`` module
    and installs a single sink on stderr that scrubs secret-like keys
    before formatting.

    Args:
        level: Minimum log level (e.g. ``"INFO"``, ``"DEBUG"``).
            Unknown values fall back to ``INFO``.

    Returns:
        A :class:`LoguruLogger` ready for ``info`` / ``warning`` /
        ``error`` calls.
    """
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    loguru_logger.configure(extra={"redacted": True})
    return LoguruLogger()


class LoguruLogger(Logger):
    """Adapter that implements :class:`Logger` against :mod:`loguru`.

    Every method is a thin wrapper that copies structured kwargs into
    loguru's bound context. The redaction step lives in the sink, so
    call sites never have to think about it.
    """

    def __init__(self) -> None:
        """Bind a private logger so tests can capture output per
        :class:`LoguruLogger` instance.
        """
        self.logger = loguru_logger.bind(component="raghub")

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an ``INFO``-level record with structured ``kwargs``.

        Args:
            message: The log message.
            **kwargs: Structured key-value pairs; secret-like keys are
                redacted by the configured sink.
        """
        self.logger.bind(**kwargs).info(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Emit a ``WARNING``-level record.

        Args:
            message: The log message.
            **kwargs: Structured key-value pairs.
        """
        self.logger.bind(**kwargs).warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an ``ERROR``-level record.

        Args:
            message: The log message.
            **kwargs: Structured key-value pairs.
        """
        self.logger.bind(**kwargs).error(message)


class LoguruSpan(Span):
    """Span whose :meth:`end` records duration into :class:`Metrics`."""

    def __init__(
        self,
        name: str,
        logger: LoguruLogger,
        metrics: Metrics,
        attributes: dict[str, Any],
    ) -> None:
        """Store the span's name and timing metadata.

        Args:
            name: Span name.
            logger: The parent :class:`LoguruLogger`.
            metrics: The metrics sink that receives the duration.
            attributes: Structured attributes attached to the span.
        """
        import time

        self.name = name
        self.logger = logger
        self.metrics = metrics
        self.attributes = attributes
        self.started = time.perf_counter()

    def end(self) -> None:
        """Record duration into the metrics sink and log completion."""
        import time

        duration_ms = (time.perf_counter() - self.started) * 1000.0
        self.metrics.record_latency(f"span.{self.name}", duration_ms, **self.attributes)
        self.logger.info(f"span.end.{self.name}", duration_ms=duration_ms, **self.attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute for later emission.

        Args:
            key: Attribute name.
            value: Attribute value.
        """
        self.attributes[key] = value


class LoguruTelemetryProvider(TelemetryProvider):
    """Telemetry provider that sinks through loguru and Prometheus."""

    def __init__(
        self,
        logger: LoguruLogger | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        """Build the provider with optional collaborators.

        Args:
            logger: The :class:`LoguruLogger` to wrap. Defaults to a
                fresh :func:`build_logger` instance.
            metrics: The metrics sink. Defaults to
                :class:`PrometheusMetrics`.
        """
        self.logger = logger or LoguruLogger()
        self.metrics = metrics or PrometheusMetrics()

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an ``info``-level record."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Emit a ``warning``-level record."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an ``error``-level record."""
        self.logger.error(message, **kwargs)

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Forward a latency observation to the metrics sink."""
        self.metrics.record_latency(name, value_ms, **labels)

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Forward a counter increment to the metrics sink."""
        self.metrics.increment(name, value, **labels)

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a new span.

        Args:
            name: Span name.
            **attrs: Attributes attached to the span.

        Returns:
            A :class:`LoguruSpan`.
        """
        return LoguruSpan(name, self.logger, self.metrics, attrs)

    def end_span(self, span: Span) -> None:
        """Close the supplied span."""
        span.end()

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage on the dedicated token counters."""
        self.metrics.increment("tokens.prompt", prompt_tokens, model=model)
        self.metrics.increment("tokens.completion", completion_tokens, model=model)
        self.logger.info(
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
            The opened span.
        """
        opened = self.start_span(name, **attrs)
        try:
            yield opened
        finally:
            self.end_span(opened)


__all__ = [
    "SECRET_KEY_RE",
    "LoguruLogger",
    "LoguruSpan",
    "LoguruTelemetryProvider",
    "build_logger",
    "redact_record",
]
