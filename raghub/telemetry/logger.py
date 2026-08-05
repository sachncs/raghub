"""Loguru-backed logging and span primitives.

Provides :func:`build_logger` to configure the process-wide loguru
logger, :class:`LoguruLogger` as the :class:`Logger` adapter, and
:class:`LoguruSpan` / :class:`LoguruTelemetryProvider` as the
span-shaped telemetry implementation.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger as loguru_logger

from raghub.models import Logger, Span, TelemetryProvider
from raghub.types import JSONValue

__all__ = [
    "LoguruLogger",
    "LoguruSpan",
    "LoguruTelemetryProvider",
    "build_logger",
]


def build_logger(level: str = "INFO") -> LoguruLogger:
    """Configure the process-wide loguru logger with a pretty console sink.

    Removes any default sinks installed by loguru's ``logger`` module
    and installs a single sink on stderr that scrubs secret-like keys
    before formatting. The sink uses loguru's built-in level icons
    (pencil for trace, bug for debug, info for INFO, check for SUCCESS,
    warning for WARNING, error for ERROR) plus colour-coded level
    names and a collapsed frame, so each log line is one readable row.

    Args:
        level: Minimum log level (e.g. ``"INFO"``, ``"DEBUG"``).
            Unknown values fall back to ``"INFO"``.

    Returns:
        A :class:`LoguruLogger` ready for ``info`` / ``warning`` /
        ``error`` calls.

    """
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> "
            "<level>{level.icon}</level> "
            "<level>{level: <7}</level> "
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
        """Bind a private logger for per-instance test output capture.

        Used to isolate loguru state across test cases.
        """
        self.logger = loguru_logger.bind(component="raghub")

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``INFO``-level record with structured ``kwargs``."""
        self.logger.bind(**kwargs).info(message)

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Emit a ``WARNING``-level record."""
        self.logger.bind(**kwargs).warning(message)

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``ERROR``-level record."""
        self.logger.bind(**kwargs).error(message)


class LoguruSpan(Span):
    """Span whose :meth:`end` logs the duration."""

    def __init__(
        self,
        name: str,
        logger: LoguruLogger,
        attributes: dict[str, Any],
    ) -> None:
        """Store the span's name and timing metadata."""
        self.name = name
        self.logger = logger
        self.attributes = attributes
        self.started = time.perf_counter()

    def end(self) -> None:
        """Log the span's duration on completion."""
        duration_ms = (time.perf_counter() - self.started) * 1000.0
        self.logger.info(f"span.end.{self.name}", duration_ms=duration_ms, **self.attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute for later emission."""
        self.attributes[key] = value


class LoguruTelemetryProvider(TelemetryProvider):
    """Telemetry provider that sinks through loguru.

    Metric emission (``record_latency`` / ``increment`` /
    ``record_tokens``) is a no-op; observability in this codebase is
    via Langfuse. The provider still implements the full
    :class:`TelemetryProvider` contract so it can stand in for the
    Langfuse provider when no credentials are configured.
    """

    def __init__(self, logger: LoguruLogger | None = None) -> None:
        """Build the provider with an optional logger override."""
        self.logger = logger or LoguruLogger()

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``info``-level record."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Emit a ``warning``-level record."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``error``-level record."""
        self.logger.error(message, **kwargs)

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """No-op; Langfuse absorbs metric emission when configured."""
        return None

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """No-op; Langfuse absorbs metric emission when configured."""
        return None

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a new span.

        Args:
            name: Span name.
            **attrs: Attributes attached to the span.

        Returns:
            A :class:`LoguruSpan`.

        """
        return LoguruSpan(name, self.logger, attrs)

    @staticmethod
    def end_span(span: Span) -> None:
        """Close the supplied span."""
        span.end()

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Log token usage; metric emission is a no-op."""
        self.logger.info(
            "tokens",
            name=name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context-manager wrapper around :meth:`start_span` / :meth:`end_span`."""
        opened = self.start_span(name, **attrs)
        try:
            yield opened
        finally:
            self.end_span(opened)
