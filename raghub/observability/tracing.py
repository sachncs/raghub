"""OpenTelemetry tracer with a console-span default exporter.

By default :class:`OpenTelemetryTracer` installs a
:class:`ConsoleSpanExporter` so spans are printed to stdout. That is
useful for local development but is rarely what you want in
production. Production deployments should call
:func:`add_otlp_exporter` (or replace ``self.processor`` /
``provider`` directly) **before** :meth:`instrument_app` so spans
are shipped to a real collector.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class OpenTelemetryTracer:
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

        Note:
            The default ``ConsoleSpanExporter`` writes spans to
            stdout. For production use, replace it with an
            ``OTLPSpanExporter`` (or another network exporter)
            before invoking :meth:`instrument_app`.
        """
        from raghub.observability.tracing_exporters import SafeConsoleSpanExporter

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(SafeConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.provider = provider
        self.tracer = trace.get_tracer(service_name)

    def instrument_app(self, app: Any) -> None:
        """Auto-instrument a FastAPI app with OpenTelemetry middleware.

        Args:
            app: A FastAPI application instance.
        """
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