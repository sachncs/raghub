"""Telemetry provider registry.

Each concrete provider registers itself with :class:`Telemetry` under
a stable name; use :meth:`Telemetry.get` for by-name dispatch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from raghub.registry import Registry
from raghub.types import JSONValue


class Span:
    """Polymorphic base for trace spans.

    Concrete spans register themselves with :class:`Telemetry`'s
    ``start_span`` method rather than here — the :class:`Telemetry`
    class owns span creation.
    """

    name: str

    def end(self) -> None:
        """End the span."""
        raise NotImplementedError

    def set_attribute(self, key: str, value: JSONValue) -> None:
        """Attach an attribute for later emission."""
        raise NotImplementedError


class Telemetry(Registry):
    """Polymorphic base for telemetry providers (logs + metrics + spans)."""

    name: str = "telemetry"

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an info-level log line."""
        raise NotImplementedError

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Emit a warning-level log line."""
        raise NotImplementedError

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an error-level log line."""
        raise NotImplementedError

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency sample under ``name`` with the given labels."""
        raise NotImplementedError

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment the ``name`` counter by ``value`` (default 1)."""
        raise NotImplementedError

    def start_span(self, name: str, **attrs: JSONValue) -> Span:
        """Start a new span and return it for later ``end_span``."""
        raise NotImplementedError

    def end_span(self, span: Span) -> None:
        """Close a previously-started span."""
        raise NotImplementedError

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage for a model call."""
        raise NotImplementedError

    @contextmanager
    def span(self, name: str, **attrs: JSONValue) -> Iterator[Span]:
        """Context-manager wrapping :meth:`start_span` / :meth:`end_span`."""
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)


__all__ = ["Span", "Telemetry"]
