"""Loguru-backed OpenTelemetry exporter.

The :class:`SafeConsoleSpanExporter` is a thin wrapper around
:class:`ConsoleSpanExporter` that survives a closed-stdout shutdown
(e.g. during pytest process shutdown). All other classes in this
module are public surface for callers that want to register their
own span processors without depending on the rest of raghub.
"""

from __future__ import annotations

from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SpanExportResult


class SafeConsoleSpanExporter(ConsoleSpanExporter):
    """Console exporter that survives a closed-stdout shutdown.

    The default :class:`ConsoleSpanExporter` raises
    :class:`ValueError` when ``sys.stdout`` is closed. That breaks
    every test that exercises a tracer and exits at process
    shutdown. This subclass wraps the ``export`` method in a guard
    that swallows the error and returns a ``FAILURE`` result.
    """

    def export(self, spans: Sequence[ReadableSpan]) -> Any:
        """Forward to the parent exporter; suppress closed-stdout errors.

        Args:
            spans: The batch of spans to export.

        Returns:
            The parent's return value (:class:`SpanExportResult.SUCCESS`
            on success) or :class:`SpanExportResult.FAILURE` on a
            closed-stdout error.
        """
        try:
            return super().export(spans)
        except ValueError as exc:
            if "closed file" in str(exc):
                return self.failed_export_result()
            raise

    def failed_export_result(self) -> Any:
        """Return :class:`SpanExportResult.FAILURE` without importing OTel types."""
        return SpanExportResult.FAILURE


__all__ = ["SafeConsoleSpanExporter", "SpanExportResult"]