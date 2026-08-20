"""Langfuse v3 telemetry adapters.

Provides the Langfuse client wrapper, :class:`LangfuseSpan`,
:class:`LangfuseTelemetryProvider`, and the convenience scorers
:func:`record_rerank_latency` and :func:`record_long_context`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from raghub.constants import ENV_LANGFUSE_PUBLIC_KEY, ENV_LANGFUSE_SECRET_KEY
from raghub.runtime import capture
from raghub.telemetry.base import Span, Telemetry
from raghub.types import JSONValue

if TYPE_CHECKING:
    pass

try:
    from langfuse import Langfuse as _Langfuse

    langfuse = _Langfuse()

    LANGFUSE_AVAILABLE = True
    IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    LANGFUSE_AVAILABLE = False
    IMPORT_ERROR = exc
    _Langfuse = Any  # pyright: ignore[reportGeneralTypeIssues]
    langfuse = Any  # pyright: ignore[reportGeneralTypeIssues]

__all__ = [
    "IMPORT_ERROR",
    "LANGFUSE_AVAILABLE",
    "Langfuse",
    "LangfuseSpan",
    "LangfuseTelemetryProvider",
    "langfuse_client",
    "record_long_context",
    "record_rerank_latency",
]

from langfuse import Langfuse  # noqa: E402


def record_rerank_latency(provider: str, seconds: float) -> None:
    """Forward a reranker latency measurement to Langfuse.

    Args:
        provider: The reranker provider name (e.g. ``"cohere"``).
        seconds: Wall-clock seconds the rerank took.

    """
    client = langfuse_client()
    if client is None:
        return
    try:
        client.score(
            name=f"raghub.rerank.{provider}.latency",
            value=seconds,
            comment=f"Rerank latency ({provider}): {seconds:.3f}s",
        )
    except Exception:
        pass


def record_long_context(*, outcome: str, seconds: float) -> None:
    """Forward a long-context summarisation measurement to Langfuse.

    Args:
        outcome: ``"ok"`` or ``"error"``.
        seconds: Wall-clock seconds the long-context step took.

    """
    client = langfuse_client()
    if client is None:
        return
    try:
        client.score(
            name="raghub.long_context",
            value=seconds,
            comment=f"Long-context ({outcome}): {seconds:.3f}s",
        )
    except Exception:
        pass


def langfuse_client() -> Any:
    """Return the active Langfuse client or ``None`` when not configured."""
    public_key = os.getenv(ENV_LANGFUSE_PUBLIC_KEY)
    secret_key = os.getenv(ENV_LANGFUSE_SECRET_KEY)
    if not public_key or not secret_key:
        return None
    try:
        return langfuse.get_client()
    except Exception:
        return None


class LangfuseSpan(Span):
    """Wrapper around a Langfuse v3 observation context."""

    def __init__(self, ctx: Any, name: str) -> None:
        """Wrap a Langfuse v3 observation ``ctx`` with a span ``name``."""
        self.ctx = ctx
        self.name = name
        self.closed = False

    def end(self) -> None:
        """Close the observation by exiting its context manager."""
        if self.closed:
            return
        self.closed = True
        exit_method = getattr(self.ctx, "__exit__", None)
        if exit_method is None:
            return
        exit_method(None, None, None)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute to the observation.

        Tries the documented ``observation.update(**)`` API first, then
        falls back to ``observation.update(metadata={...})``.
        """
        update = getattr(self.ctx, "update", None)
        if update is None:
            return
        update(**{key: value})


@Telemetry.register("langfuse")
class LangfuseTelemetryProvider(Telemetry):
    """Langfuse-backed telemetry provider.

    Implements the full :class:`TelemetryProvider` contract:
    logging (``info``/``warning``/``error``), metrics
    (``record_latency``/``increment``), spans (``start_span``/
    ``end_span``), and token tracking (``record_tokens``).

    When ``langfuse`` is not installed or no credentials are
    configured, every method silently no-ops so the framework keeps
    working without telemetry.
    """

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        flush_interval: float = 1.0,
    ) -> None:
        """Initialise the provider.

        Args:
            public_key: Langfuse public key (defaults to env).
            secret_key: Langfuse secret key (defaults to env).
            host: Langfuse host URL.
            flush_interval: Seconds between background flushes.

        """
        public_key = public_key or os.getenv(ENV_LANGFUSE_PUBLIC_KEY)
        secret_key = secret_key or os.getenv(ENV_LANGFUSE_SECRET_KEY)
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self.flush_interval = flush_interval
        self.client: Any = None
        if LANGFUSE_AVAILABLE and public_key and secret_key:
            self.client = self.langfuse_client(host, public_key, secret_key, flush_interval)

    @staticmethod
    def is_configured() -> bool:
        """Return ``True`` when Langfuse credentials are present in env.

        Returns:
            ``True`` if the ``langfuse`` package is installed and both
            ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are
            set in the environment.

        """
        return bool(
            LANGFUSE_AVAILABLE
            and os.getenv(ENV_LANGFUSE_PUBLIC_KEY)
            and os.getenv(ENV_LANGFUSE_SECRET_KEY)
        )

    @staticmethod
    def langfuse_client(
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        flush_interval: float = 1.0,
    ) -> Any:
        """Create and return a Langfuse v3 client.

        Args:
            host: Langfuse host URL (defaults to env).
            public_key: Langfuse public key (defaults to env).
            secret_key: Langfuse secret key (defaults to env).
            flush_interval: Seconds between background flushes.

        Returns:
            A configured Langfuse client instance.

        """
        return _Langfuse(
            public_key=public_key or os.getenv(ENV_LANGFUSE_PUBLIC_KEY),
            secret_key=secret_key or os.getenv(ENV_LANGFUSE_SECRET_KEY),
            host=host,
            flush_interval=flush_interval,
        )

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Log an info-level event.

        Args:
            message: Human-readable log message.
            **kwargs: Arbitrary metadata forwarded to Langfuse.

        """
        if self.client is None:
            return
        try:
            self.client.event(name="info", input={"message": message, **kwargs})
        except Exception:
            pass

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Log a warning-level event.

        Args:
            message: Human-readable warning message.
            **kwargs: Arbitrary metadata forwarded to Langfuse.

        """
        if self.client is None:
            return
        try:
            self.client.event(name="warning", input={"message": message, **kwargs})
        except Exception:
            pass

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """Log an error-level event.

        Args:
            message: Human-readable error message.
            **kwargs: Arbitrary metadata forwarded to Langfuse.

        """
        if self.client is None:
            return
        try:
            self.client.event(name="error", input={"message": message, **kwargs})
        except Exception:
            pass

    def record_latency(self, name: str, seconds: float, **kwargs: JSONValue) -> None:
        """Record a latency measurement.

        Args:
            name: Metric name (e.g. ``"raghub.query.total"``).
            seconds: Wall-clock seconds.
            **kwargs: Arbitrary metadata forwarded to Langfuse.

        """
        if self.client is None:
            return
        try:
            self.client.score(name=name, value=seconds, comment=f"{name}: {seconds:.3f}s")
        except Exception:
            pass

    def increment(self, name: str, value: int = 1, **kwargs: JSONValue) -> None:
        """Increment a counter.

        Args:
            name: Metric name.
            value: Delta (default 1).
            **kwargs: Arbitrary metadata forwarded to Langfuse.

        """
        if self.client is None:
            return
        try:
            self.client.score(name=name, value=value, comment=f"{name} +{value}")
        except Exception:
            pass

    def start_span(self, name: str, **attrs: Any) -> LangfuseSpan:
        """Start a new span.

        Args:
            name: Span name.
            **attrs: Arbitrary attributes forwarded to Langfuse.

        Returns:
            A :class:`LangfuseSpan` wrapping the Langfuse observation.

        """
        if self.client is None:
            return LangfuseSpan(Any, name)  # pyright: ignore[reportGeneralTypeIssues]
        ctx = self.client.span(name=name, metadata=attrs)
        return LangfuseSpan(ctx, name)

    def end_span(self, span: Span) -> None:
        """End a span.

        Args:
            span: The span to close.

        """
        if isinstance(span, LangfuseSpan):
            span.end()

    def record_tokens(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
        **kwargs: Any,
    ) -> None:
        """Record token usage.

        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            model: Model name.
            **kwargs: Arbitrary metadata forwarded to Langfuse.

        """
        if self.client is None:
            return
        try:
            self.client.score(
                name="raghub.tokens",
                value=prompt_tokens + completion_tokens,
                usage_details={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                }
            )
        except Exception:
            pass

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Any]:
        """Context-manager wrapper around :meth:`start_span` / :meth:`end_span`."""
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)
