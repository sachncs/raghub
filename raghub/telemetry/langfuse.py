"""Langfuse v3 telemetry provider.

A drop-in :class:`raghub.interfaces.observability.TelemetryProvider`
adapter that ships events to Langfuse. When ``langfuse`` is not
installed or no credentials are configured, the provider still
implements the contract but no-ops every method, so the framework
keeps working without telemetry.

Uses the Langfuse v3 SDK (``get_client()`` and
``start_as_current_observation``). When a v2 SDK is detected at
import time, the provider falls back to the legacy v2 API while
remaining syntactically compatible with the new contract.

Every public method is wrapped in :meth:`_safe`, a small helper that
swallows exceptions. Telemetry must never break the host application,
so any failure inside Langfuse is dropped silently. Operators who
need to debug telemetry failures should enable the ``LANGFUSE_DEBUG``
environment variable; the provider logs the failure in that case.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from raghub.interfaces.observability import Span, TelemetryProvider

T = TypeVar("T")

langfuse_get_client: Any
LangfuseLegacy: Any

try:
    from langfuse import Langfuse as LangfuseLegacy
    from langfuse import get_client as langfuse_get_client

    LANGFUSE_AVAILABLE = True
    IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    langfuse_get_client = None
    LangfuseLegacy = None
    LANGFUSE_AVAILABLE = False
    IMPORT_ERROR = exc

LOGGER = logging.getLogger("raghub.telemetry.langfuse")


class NoopSpan(Span):
    """No-op span implementation.

    Used when Langfuse is not installed or no credentials are
    configured. Implements the :class:`Span` protocol so callers can
    treat it interchangeably with the live implementation.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict[str, Any] = {}

    def end(self) -> None:
        """No-op."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Capture an attribute for any finaliser to read."""
        self.attrs[key] = value

    @property
    def attributes(self) -> dict[str, Any]:
        """Return the attributes attached to this span."""
        return dict(self.attrs)


class LangfuseSpan(Span):
    """Wrapper around a Langfuse v3 observation context."""

    def __init__(self, ctx: Any, name: str) -> None:
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
        try:
            update(**{key: value})
        except Exception:
            try:
                update(metadata={key: value})
            except Exception:
                pass


class LangfuseTelemetryProvider(TelemetryProvider):
    """Langfuse-backed telemetry provider.

    Implements the full :class:`TelemetryProvider` contract:
    logging (``info``/``warning``/``error``), metrics
    (``record_latency``/``increment``), spans (``start_span``/
    ``end_span``), and token tracking (``record_tokens``).
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
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self.flush_interval = flush_interval
        self.client: Any = None
        if LANGFUSE_AVAILABLE and public_key and secret_key:
            self.client = self.safe_call(
                self.build_langfuse_client,
                host,
                public_key,
                secret_key,
                flush_interval,
            )

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
            and os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
        )

    @staticmethod
    def safe_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
        """Run ``fn``; swallow exceptions and log them when ``LANGFUSE_DEBUG`` is set.

        Telemetry must never crash the host application. Every public
        method of this provider routes through this helper.

        Args:
            fn: The callable to invoke.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The callable's return value, or ``None`` if it raised.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if os.getenv("LANGFUSE_DEBUG"):
                LOGGER.warning("langfuse telemetry failure: %s", exc)
            return None

    def build_langfuse_client(
        self, host: str | None, public_key: str, secret_key: str, flush_interval: float
    ) -> Any:
        """Build a v3 client if available, else fall back to v2.

        Args:
            host: Langfuse host URL.
            public_key: Public key.
            secret_key: Secret key.
            flush_interval: Background flush interval in seconds.

        Returns:
            A Langfuse client instance, or ``None`` when neither v3
            nor v2 SDKs are available.
        """
        if langfuse_get_client is not None:
            try:
                return langfuse_get_client()
            except Exception:
                pass
        if LangfuseLegacy is not None:
            return LangfuseLegacy(
                public_key=public_key,
                secret_key=secret_key,
                host=host or "https://cloud.langfuse.com",
                flush_interval=flush_interval,
            )
        return None

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an info log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.
        """
        with self.span(f"log.info.{message}", level="info", **kwargs):
            pass

    def warning(self, message: str, **kwargs: Any) -> None:
        """Emit a warning log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.
        """
        with self.span(f"log.warning.{message}", level="warning", **kwargs):
            pass

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an error log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.
        """
        with self.span(f"log.error.{message}", level="error", **kwargs):
            pass

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency via a span.

        Args:
            name: Metric name.
            value_ms: Latency in milliseconds.
            **labels: Optional label set.
        """
        with self.span(f"latency.{name}", value_ms=value_ms, **labels):
            pass

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment a counter via a span.

        Args:
            name: Counter name.
            value: Increment amount.
            **labels: Optional label set.
        """
        with self.span(f"counter.{name}", increment=value, **labels):
            pass

    # ------------------------------------------------------------------
    # Spans
    # ------------------------------------------------------------------

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a Langfuse span (v3) or fall back to a no-op span.

        Args:
            name: Span name.
            **attrs: Span attributes. ``user_id`` and ``session_id``
                are propagated to every child observation via
                ``propagate_attributes`` so Langfuse traces carry
                the multi-user attribution.

        Returns:
            A :class:`Span` (live or no-op).
        """
        if self.client is None:
            return NoopSpan(name)
        # Propagate user/session attributes to child observations.
        propagate = {k: v for k, v in attrs.items() if k in ("user_id", "session_id") and v}
        if propagate:
            self.safe_call(self.propagate_to_langfuse, **propagate)
        start_obs = getattr(self.client, "start_as_current_observation", None)
        if start_obs is None:
            return NoopSpan(name)
        ctx = self.safe_call(start_obs, as_type="span", name=name, **{"input": attrs})
        if ctx is None:
            return NoopSpan(name)
        return LangfuseSpan(ctx, name)

    def propagate_to_langfuse(self, **attrs: Any) -> None:
        """Call Langfuse ``propagate_attributes`` if available.

        Args:
            **attrs: Key/value pairs to attach to every subsequent
                observation on the current thread.
        """
        propagate = getattr(self.client, "propagate_attributes", None)
        if propagate is not None:
            propagate(**attrs)

    def end_span(self, span: Span) -> None:
        """Close a previously-opened span.

        Args:
            span: The span returned by :meth:`start_span`.
        """
        if isinstance(span, NoopSpan):
            return
        self.safe_call(span.end)

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage as a Langfuse generation.

        Args:
            name: Generation name.
            prompt_tokens: Input token count.
            completion_tokens: Output token count.
            model: Model identifier.
        """
        if self.client is None:
            return
        start_obs = getattr(self.client, "start_as_current_observation", None)
        if start_obs is None:
            return
        gen = self.safe_call(start_obs, as_type="generation", name=name, model=model)
        if gen is None:
            return

        @contextmanager
        def ctx() -> Iterator[Any]:
            try:
                with gen:
                    yield gen
            except Exception:
                pass

        with ctx() as cm:
            if cm is not None:
                try:
                    cm.update(
                        usage_details={
                            "input": prompt_tokens,
                            "output": completion_tokens,
                        }
                    )
                except Exception:
                    pass

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context-manager wrapper around :meth:`start_span` / :meth:`end_span`.

        Args:
            name: Span name.
            **attrs: Span attributes.

        Yields:
            The open :class:`Span`.
        """
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)

    # ------------------------------------------------------------------
    # Backwards-compat shim
    # ------------------------------------------------------------------

    def end_trace(self) -> None:
        """Flush buffered events; kept for backwards compatibility."""
        if self.client is None:
            return
        flush = getattr(self.client, "flush", None)
        if flush is None:
            return
        self.safe_call(flush)


__all__ = ["LangfuseSpan", "LangfuseTelemetryProvider", "NoopSpan"]
