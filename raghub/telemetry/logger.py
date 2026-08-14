"""Loguru-backed logging and span primitives.

Uses :mod:`loguru` directly for :class:`Logger` and span emission;
the :class:`TelemetryProvider` implementation wires loguru's
``logger`` to the framework's structured-kwargs contract.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

from raghub.telemetry.base import Span, Telemetry
from raghub.types import JSONValue

__all__ = [
    "Logger",
    "LoguruSpan",
    "LoguruTelemetryProvider",
    "logger",
]


class Logger:
    """Adapter that implements the logger contract against :mod:`loguru`.

    Each call binds structured kwargs into the loguru record so they
    appear in the formatted output.
    """

    def __init__(self) -> None:
        """Bind a logger instance for structured output."""
        self.bound = logger.bind(component="raghub")

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``INFO``-level record with structured ``kwargs``."""
        self.bound.bind(**kwargs).info(message)

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Emit a ``WARNING``-level record with structured ``kwargs``."""
        self.bound.bind(**kwargs).warning(message)

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``ERROR``-level record with structured ``kwargs``."""
        self.bound.bind(**kwargs).error(message)


class LoguruSpan(Span):
    """Span whose :meth:`end` logs the duration via loguru."""

    def __init__(
        self,
        name: str,
        adapter: Logger,
        attributes: dict[str, Any],
    ) -> None:
        """Store the span's name and timing metadata."""
        self.name = name
        self.adapter = adapter
        self.attributes = attributes
        self.started = time.perf_counter()

    def end(self) -> None:
        """Log the span's duration on completion."""
        duration_ms = (time.perf_counter() - self.started) * 1000.0
        self.adapter.info(f"span.end.{self.name}", duration_ms=duration_ms, **self.attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute for later emission."""
        self.attributes[key] = value


@Telemetry.register("loguru")
class LoguruTelemetryProvider(Telemetry):
    """Telemetry provider that sinks through :mod:`loguru`."""

    name = "loguru"

    def __init__(self, adapter: Logger | None = None) -> None:
        """Build the provider with an optional adapter override."""
        self.adapter = adapter or Logger()

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``info``-level record."""
        self.adapter.info(message, **kwargs)

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Emit a ``warning``-level record."""
        self.adapter.warning(message, **kwargs)

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an ``error``-level record."""
        self.adapter.error(message, **kwargs)

    @staticmethod
    def record_latency(name: str, value_ms: float, **labels: Any) -> None:
        """No-op; Langfuse absorbs metric emission when configured."""

    @staticmethod
    def increment(name: str, value: int = 1, **labels: Any) -> None:
        """No-op; Langfuse absorbs metric emission when configured."""

    def start_span(self, name: str, **attrs: Any) -> LoguruSpan:
        """Open a new :class:`LoguruSpan`."""
        return LoguruSpan(name, self.adapter, attrs)

    @staticmethod
    def end_span(span: LoguruSpan) -> None:
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
        self.adapter.info(
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
