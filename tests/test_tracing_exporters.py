"""Unit tests for :class:`SafeConsoleSpanExporter`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from raghub.observability.tracing_exporters import SafeConsoleSpanExporter


def _stub_span() -> MagicMock:
    return MagicMock()


def test_export_passes_through_success_result() -> None:
    """A successful parent ``export`` returns ``SpanExportResult.SUCCESS``."""
    exporter = SafeConsoleSpanExporter()
    spans = [_stub_span()]
    with patch(
        "opentelemetry.sdk.trace.export.ConsoleSpanExporter.export",
        return_value=SpanExportResult.SUCCESS,
    ):
        assert exporter.export(spans) == SpanExportResult.SUCCESS


def test_export_reraises_unrelated_value_errors() -> None:
    """``ValueError`` messages that don't mention a closed file are re-raised."""
    exporter = SafeConsoleSpanExporter()
    with patch(
        "opentelemetry.sdk.trace.export.ConsoleSpanExporter.export",
        side_effect=ValueError("some other problem"),
    ):
        with pytest.raises(ValueError, match="some other problem"):
            exporter.export([_stub_span()])


def test_export_returns_failure_on_closed_stdout() -> None:
    """A ``ValueError`` referencing a closed file returns a failure result."""
    exporter = SafeConsoleSpanExporter()
    with patch(
        "opentelemetry.sdk.trace.export.ConsoleSpanExporter.export",
        side_effect=ValueError("I/O operation on closed file"),
    ):
        result = exporter.export([_stub_span()])
    assert result == SpanExportResult.FAILURE


def test_failed_export_result_returns_failure() -> None:
    """``failed_export_result`` exposes a stable failure sentinel."""
    exporter = SafeConsoleSpanExporter()
    assert exporter.failed_export_result() == SpanExportResult.FAILURE


def test_export_with_empty_span_list() -> None:
    """An empty span list is forwarded to the parent exporter."""
    exporter = SafeConsoleSpanExporter()
    with patch(
        "opentelemetry.sdk.trace.export.ConsoleSpanExporter.export",
        return_value=SpanExportResult.SUCCESS,
    ) as mock_export:
        assert exporter.export([]) == SpanExportResult.SUCCESS
        mock_export.assert_called_once_with([])
