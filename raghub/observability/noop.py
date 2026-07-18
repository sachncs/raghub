"""No-op telemetry provider.

The default :class:`raghub.interfaces.observability.TelemetryProvider`
when no remote backend is configured. Every method is a no-op so the
framework runs without emitting any observability data.

The :class:`NoopSpan` class is also re-exported here for callers
that need a Span-conforming sentinel without the overhead of
constructing a real one.
"""

from __future__ import annotations

from typing import Any

from raghub.interfaces.observability import Span, TelemetryProvider
from raghub.telemetry.langfuse import NoopSpan

__all__ = ["NoOpTelemetry", "NoopSpan"]


class NoOpTelemetry(TelemetryProvider):
    """Silent telemetry provider; satisfies the contract."""

    def info(self, message: str, **kwargs: Any) -> None:
        """No-op."""

    def warning(self, message: str, **kwargs: Any) -> None:
        """No-op."""

    def error(self, message: str, **kwargs: Any) -> None:
        """No-op."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """No-op."""

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """No-op."""

    def start_span(self, name: str, **attrs: Any) -> Span:
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
