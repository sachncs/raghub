from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


class OpenTelemetryTracer:
    def __init__(self, service_name: str = "raghub") -> None:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)

    def instrument_app(self, app: Any) -> None:
        FastAPIInstrumentor.instrument_app(app)

    def create_span(self, name: str) -> Any:
        return self.tracer.start_as_current_span(name)
