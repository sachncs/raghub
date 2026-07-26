"""Observability helpers: no-op, redacting, and loguru-backed implementations.

The module ships three reusable adapters:

* :class:`NoOpTelemetry` — silent default; satisfies the
  :class:`raghub.interfaces.observability.TelemetryProvider` contract
  without performing any I/O.
* :class:`RedactingTelemetry` — wraps another telemetry provider and
  scrubs kwargs whose keys look like secrets before forwarding.
* :class:`LoguruTelemetryProvider` — adapts the new contract to the
  loguru + Prometheus + OpenTelemetry stack.
"""

from raghub.observability.logging import LoguruLogger, LoguruTelemetryProvider, build_logger
from raghub.observability.noop import NoOpTelemetry
from raghub.observability.redact import RedactingTelemetry

__all__ = [
    "LoguruLogger",
    "LoguruTelemetryProvider",
    "NoOpTelemetry",
    "RedactingTelemetry",
    "build_logger",
]