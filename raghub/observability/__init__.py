"""Observability helpers: no-op, redacting, and structural implementations.

The module ships three reusable adapters:

* :class:`NoOpTelemetry` — silent default; satisfies the
  :class:`raghub.interfaces.observability.TelemetryProvider` contract
  without performing any I/O.
* :class:`RedactingTelemetry` — wraps another telemetry provider and
  scrubs kwargs whose keys look like secrets before forwarding.
* :class:`StructlogTelemetryProvider` — adapts the legacy structlog +
  Prometheus + OTel stack to the new contract.
"""

from raghub.observability.noop import NoOpTelemetry
from raghub.observability.redact import RedactingTelemetry
from raghub.observability.structlog_provider import StructlogTelemetryProvider

__all__ = [
    "NoOpTelemetry",
    "RedactingTelemetry",
    "StructlogTelemetryProvider",
]
