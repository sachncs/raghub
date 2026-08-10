"""Observability and telemetry surface.

The framework ships three interoperable telemetry layers split across
focused submodules:

* :mod:`raghub.telemetry.tracing` — v3 Langfuse adapter and
  OpenTelemetry :class:`Tracer` (the spec default).
* :mod:`raghub.telemetry.logger` — loguru adapter for users who prefer
  in-process logging without a remote service.
* :mod:`raghub.telemetry.metrics` — silent
  :class:`NoOpTelemetry` / :class:`NoopSpan` satisfying the
  :class:`TelemetryProvider` contract with zero I/O.
* :mod:`raghub.telemetry.redaction` — :class:`RedactingTelemetry`
  wrapping any provider and scrubbing kwargs whose keys look like
  secrets before forwarding.

Public constants:

* :data:`SECRET_KEY_RE` — the regex used by the redacting layer.
"""

from __future__ import annotations

from raghub.telemetry.logger import (
    Logger,
    LoguruSpan,
    LoguruTelemetryProvider,
)
from raghub.telemetry.metrics import (
    NoopSpan,
    NoOpTelemetry,
)
from raghub.telemetry.redaction import (
    SECRET_KEY_RE,
    RedactingTelemetry,
    redact_record,
    scrub_secrets,
)
from raghub.telemetry.tracing import (
    LANGFUSE_AVAILABLE,
    Langfuse,
    LangfuseSpan,
    LangfuseTelemetryProvider,
    SafeConsoleSpanExporter,
    Tracer,
    langfuse_client,
    langfuse_get_client,
    record_long_context,
    record_rerank_latency,
    try_import_submodule,
)

__all__ = [
    "LANGFUSE_AVAILABLE",
    "SECRET_KEY_RE",
    "Langfuse",
    "LangfuseSpan",
    "LangfuseTelemetryProvider",
    "Logger",
    "LoguruSpan",
    "LoguruTelemetryProvider",
    "NoOpTelemetry",
    "NoopSpan",
    "RedactingTelemetry",
    "SafeConsoleSpanExporter",
    "Tracer",
    "langfuse_client",
    "langfuse_get_client",
    "record_long_context",
    "record_rerank_latency",
    "redact_record",
    "scrub_secrets",
    "try_import_submodule",
]