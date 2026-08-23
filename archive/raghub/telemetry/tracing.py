"""Distributed-tracing primitives: Langfuse and OpenTelemetry.

Hosts the Langfuse v3 client wrappers (:class:`LangfuseSpan`,
:class:`LangfuseTelemetryProvider`), the convenience scorers
(:func:`record_rerank_latency`, :func:`record_long_context`),
and the OpenTelemetry :class:`Tracer` with its closed-stdout-safe
:class:`SafeConsoleSpanExporter`.

.. seealso:: :mod:`raghub.telemetry.langfuse`
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raghub.errors import ConfigurationError
from raghub.runtime import capture
from raghub.telemetry.base import Span, Telemetry
from raghub.telemetry.langfuse import (  # re-export for backward compat
    IMPORT_ERROR,
    LANGFUSE_AVAILABLE,
    Langfuse,
    LangfuseSpan,
    LangfuseTelemetryProvider,
    langfuse_client,
    record_long_context,
    record_rerank_latency,
)
from raghub.types import JSONValue

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExportResult

__all__ = [
    "IMPORT_ERROR",
    "LANGFUSE_AVAILABLE",
    "Langfuse",
    "LangfuseSpan",
    "LangfuseTelemetryProvider",
    "SafeConsoleSpanExporter",
    "Tracer",
    "langfuse_client",
    "record_long_context",
    "record_rerank_latency",
]


class SafeConsoleSpanExporter:
    """Console exporter that survives a closed-stdout shutdown.

    The default :class:`ConsoleSpanExporter` raises
    :class:`ValueError` when ``sys.stdout`` is closed. That breaks
    every test that exercises a tracer and exits at process
    shutdown. This subclass wraps the ``export`` method in a guard
    that swallows the error and returns a ``FAILURE`` result.
    """

    def __init__(self, *args: Any, **kwargs: JSONValue) -> None:
        """Lazy-import the parent :class:`ConsoleSpanExporter`."""
        self.parent = ConsoleSpanExporter(*args, **kwargs)

    def export(self, spans: Sequence[Any]) -> Any:
        """Forward to the parent exporter; suppress closed-stdout errors.

        Args:
            spans: The batch of spans to export.

        Returns:
            The parent's return value or ``SpanExportResult.FAILURE``
            on a closed-stdout error.

        """
        result, error = capture(self.parent.export, spans)
        if error is None:
            return result
        if "closed file" in str(error):
            return self.failure()
        raise error

    def shutdown(self) -> None:
        """Forward shutdown to the parent exporter."""
        self.parent.shutdown()

    @staticmethod
    def failure() -> Any:
        """Return the OTel FAILURE result."""
        return SpanExportResult.FAILURE


class Tracer:
    """Wrap an OpenTelemetry tracer provider with FastAPI auto-instrumentation.

    Attributes:
        provider: The underlying :class:`TracerProvider`. Exposed so
            callers can swap processors / exporters before
            :meth:`instrument_app` runs.
        tracer: The :class:`trace.Tracer` instance used to create
            spans manually.

    """

    def __init__(self, service_name: str = "raghub") -> None:
        """Configure a tracer provider with a console span exporter.

        Args:
            service_name: The ``service.name`` resource attribute.

        Raises:
            ConfigurationError: When OpenTelemetry SDK packages are
                not installed.

        """
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(SafeConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.provider = provider
        self.tracer = trace.get_tracer(service_name)

    def add_exporter(self, *, endpoint: str, insecure: bool = True) -> None:
        """Replace the default console exporter with an OTLP one.

        Args:
            endpoint: The OTLP collector endpoint (e.g.
                ``"http://otel-collector:4317"``).
            insecure: When ``True`` (default) use HTTP/gRPC without
                TLS. Production deployments should set this to
                ``False`` and supply a TLS endpoint.

        """
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        self.provider.add_span_processor(BatchSpanProcessor(exporter))

    @staticmethod
    def instrument_app(app: Any) -> None:
        """Auto-instrument a FastAPI app with OpenTelemetry middleware.

        Args:
            app: A FastAPI application instance.

        """
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)

    def create_span(self, name: str) -> Any:
        """Open a new span as a context manager.

        Args:
            name: The span name.

        Returns:
            The :class:`opentelemetry.trace.Span` context manager
            from :meth:`tracer.start_as_current_span`.

        """
        return self.tracer.start_as_current_span(name)

    def shutdown(self) -> None:
        """Flush and shut down the underlying provider.

        Safe to call multiple times.
        """
        capture(self.provider.shutdown)
