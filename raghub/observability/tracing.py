"""Loguru-backed OpenTelemetry tracer.

The :class:`Tracer` wraps :mod:`opentelemetry.sdk.trace` so spans can
be either printed to stdout (development) or shipped over OTLP
(production). The public surface is small and stable:

* :meth:`Tracer.add_otlp_exporter` — swap the default console exporter
  for an OTLP one; safe to call once during application startup.
* :meth:`Tracer.shutdown` — flush and shut down the underlying
  provider.
* :meth:`Tracer.instrument_app` — install FastAPI auto-instrumentation.

Example:
    >>> from raghub.observability.tracing import Tracer
    >>> tracer = Tracer()
    >>> tracer.add_otlp_exporter(endpoint="http://collector:4317")
"""

from __future__ import annotations

from typing import Any

from raghub.exceptions import ConfigurationError


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
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            from raghub.observability.tracing_exporters import SafeConsoleSpanExporter
        except ImportError as exc:
            raise ConfigurationError("OpenTelemetry tracing requires opentelemetry-sdk") from exc

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(SafeConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.provider = provider
        self.tracer = trace.get_tracer(service_name)

    def add_otlp_exporter(self, *, endpoint: str, insecure: bool = True) -> None:
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

    def instrument_app(self, app: Any) -> None:
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
        try:
            self.provider.shutdown()
        except Exception:
            # Shutdown is best-effort; an already-shut-down provider
            # would otherwise raise a confusing error during tests.
            return


__all__ = ["Tracer"]
