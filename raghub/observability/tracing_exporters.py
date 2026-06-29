"""OpenTelemetry tracing helpers.

Defines :class:`SafeConsoleSpanExporter`, a small wrapper around
:class:`ConsoleSpanExporter` that swallows :class:`ValueError`
exceptions raised when stdout is closed (e.g. during pytest's
process shutdown).
"""

from __future__ import annotations

from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import ConsoleSpanExporter


class SafeConsoleSpanExporter(ConsoleSpanExporter):
    """Console exporter that survives a closed-stdout shutdown.

    The default :class:`ConsoleSpanExporter` raises
    :class:`ValueError` when ``sys.stdout`` is closed. That breaks
    every test that exercises a tracer and exits at process
    shutdown. This subclass wraps the ``export`` method in a guard
    that swallows the error.
    """

    def export(self, spans: Sequence[ReadableSpan]) -> Any:
        """Forward to the parent exporter; suppress closed-stdout errors.

        Args:
            spans: The batch of spans to export.

        Returns:
            The parent's return value (``SpanExportResult.SUCCESS``
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
        """Return a SpanExportResult.FAILURE without importing OTel types."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.FAILURE
