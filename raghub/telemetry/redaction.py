"""Secret-redaction primitives for telemetry payloads.

Provides :data:`SECRET_KEY_RE` (the regex used to detect secret-shaped
keys), :func:`scrub_secrets` (returns a copy of a dict with masked
values), :func:`redact_record` (in-place redaction of a loguru record),
and :class:`RedactingTelemetry` (a :class:`TelemetryProvider` wrapper
that scrubs kwargs before forwarding).
"""

from __future__ import annotations

import re
from typing import Any

from raghub.models import Span, TelemetryProvider
from raghub.types import JSONValue

__all__ = [
    "RedactingTelemetry",
    "SECRET_KEY_RE",
    "redact_record",
    "scrub_secrets",
]


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


def redact_record(record: dict[str, Any]) -> None:
    """In-place redact secret-looking values in a loguru record.

    Args:
        record: The mutable loguru ``record.message`` dictionary; values
            whose key matches :data:`SECRET_KEY_RE` are replaced by
            ``"***"``. Nested dicts and lists are scrubbed recursively;
            keys at every depth are checked against the secret pattern.

    """

    def scrub(value: Any) -> Any:
        """Recursively mask any dict value whose key matches the secret pattern."""
        if isinstance(value, dict):
            scrubbed: dict[str, Any] = {}
            for key, inner in value.items():
                if SECRET_KEY_RE.search(str(key)):
                    scrubbed[key] = "***"
                else:
                    scrubbed[key] = scrub(inner)
            return scrubbed
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    scrubbed = scrub(record)
    record.clear()
    record.update(scrubbed)


class RedactingTelemetry(TelemetryProvider):
    """Telemetry wrapper that redacts secret-looking keys."""

    def __init__(self, inner: TelemetryProvider) -> None:
        """Wrap ``inner`` with secret-redaction.

        Args:
            inner: The downstream provider receiving redacted calls.

        """
        self.inner = inner

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Forward ``info`` with redacted kwargs."""
        self.inner.info(message, **scrub_secrets(kwargs))

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Forward ``warning`` with redacted kwargs."""
        self.inner.warning(message, **scrub_secrets(kwargs))

    def error(self, message: str, **kwargs: JSONValue) -> None:
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
