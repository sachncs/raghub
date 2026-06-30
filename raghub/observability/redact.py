"""Redacting telemetry wrapper.

Filters kwargs whose keys look like secrets (``password``,
``api_key``, ``token``, ``jwt``, ``secret``, …) before forwarding
to the underlying telemetry provider. This is the last-mile defence
against accidental secret leakage through log messages.
"""

from __future__ import annotations

import re
from typing import Any

from raghub.interfaces.observability import Span, TelemetryProvider

SECRET_KEY_RE = re.compile(
    r"(?i)(password|passwd|secret|api_key|apikey|access_token|refresh_token|jwt|authorization)"
)


def scrub_secrets(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``kwargs`` with secret-looking values masked."""
    scrubbed: dict[str, Any] = {}
    for key, value in kwargs.items():
        if SECRET_KEY_RE.search(key):
            scrubbed[key] = "***"
        elif isinstance(value, dict):
            scrubbed[key] = scrub_secrets(value)
        else:
            scrubbed[key] = value
    return scrubbed


class RedactingTelemetry(TelemetryProvider):
    """Telemetry wrapper that redacts secret-looking keys."""

    def __init__(self, inner: TelemetryProvider) -> None:
        """Wrap ``inner`` with secret-redaction."""
        self.inner = inner

    def info(self, message: str, **kwargs: Any) -> None:
        """Forward ``info`` with redacted kwargs."""
        self.inner.info(message, **scrub_secrets(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Forward ``warning`` with redacted kwargs."""
        self.inner.warning(message, **scrub_secrets(kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Forward ``error`` with redacted kwargs."""
        self.inner.error(message, **scrub_secrets(kwargs))

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Forward ``record_latency`` with redacted labels."""
        self.inner.record_latency(name, value_ms, **scrub_secrets(labels))

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Forward ``increment`` with redacted labels."""
        self.inner.increment(name, value, **scrub_secrets(labels))

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Forward ``start_span`` with redacted attributes."""
        return self.inner.start_span(name, **scrub_secrets(attrs))

    def end_span(self, span: Span) -> None:
        """Forward ``end_span``."""
        self.inner.end_span(span)

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Forward ``record_tokens``."""
        self.inner.record_tokens(name, prompt_tokens, completion_tokens, model)


__all__ = ["RedactingTelemetry"]
