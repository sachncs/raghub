"""No-op telemetry primitives.

Provides :class:`NoopSpan` and :class:`NoOpTelemetry` for environments
that should not emit any telemetry. Both implement the
:class:`TelemetryProvider` contract so they can be used interchangeably
with the live providers.
"""

from __future__ import annotations

from typing import Any

from raghub.models import Span, TelemetryProvider
from raghub.types import JSONValue

__all__ = [
    "NoOpTelemetry",
    "NoopSpan",
]


class NoopSpan(Span):
    """No-op span implementation.

    Used when Langfuse is not installed or no credentials are
    configured. Implements the :class:`Span` protocol so callers can
    treat it interchangeably with the live implementation.
    """

    def __init__(self, name: str) -> None:
        """Store the span name; no exporter is wired."""
        self.name = name
        self.attrs: dict[str, Any] = {}

    def end(self) -> None:
        """No-op."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Capture an attribute for any finaliser to read."""
        self.attrs[key] = value

    @property
    def attributes(self) -> dict[str, Any]:
        """Return the attributes attached to this span."""
        return dict(self.attrs)


class NoOpTelemetry(TelemetryProvider):
    """Silent telemetry provider; satisfies the contract."""

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """No-op."""

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """No-op."""

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """No-op."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """No-op."""

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """No-op."""

    @staticmethod
    def start_span(name: str, **attrs: Any) -> Span:
        """Return a no-op span.

        Args:
            name: Span name (recorded for completeness).
            **attrs: Span attributes (ignored).

        Returns:
            A :class:`NoopSpan`.

        """
        return NoopSpan(name)

    def end_span(self, span: Span) -> None:
        """No-op."""

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """No-op."""
