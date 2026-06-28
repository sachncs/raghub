"""Logging, metrics, and tracing helpers.

This package wraps third-party observability primitives
(:mod:`structlog`, :mod:`prometheus_client`, :mod:`opentelemetry`)
behind thin adapters so the rest of the codebase doesn't import them
directly. Tests can swap in NullMetrics without affecting other
components.
"""

from .logging import StructuredLogger, build_logger
from .metrics import NullMetrics, PrometheusMetrics
from .tracing import OpenTelemetryTracer

__all__ = [
    "NullMetrics",
    "OpenTelemetryTracer",
    "PrometheusMetrics",
    "StructuredLogger",
    "build_logger",
]