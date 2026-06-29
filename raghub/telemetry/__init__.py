"""Telemetry / observability adapters.

The default adapter is Langfuse (spec requirement). The previous
in-process stack (structlog + Prometheus + OpenTelemetry) is kept
under :mod:`raghub.observability` for users who prefer not to send
data to a third-party service.

Public re-exports:

* :class:`LangfuseTelemetryProvider` — the v3 Langfuse adapter.
* :class:`LangfuseSpan` / :class:`NoopSpan` — concrete span types.
"""

from raghub.telemetry.langfuse import (
    LangfuseSpan,
    LangfuseTelemetryProvider,
    NoopSpan,
)

__all__ = ["LangfuseSpan", "LangfuseTelemetryProvider", "NoopSpan"]

