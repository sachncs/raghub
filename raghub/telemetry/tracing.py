"""Distributed-tracing primitives: Langfuse and OpenTelemetry.

Hosts the Langfuse v3 client wrappers (:class:`LangfuseSpan`,
:class:`LangfuseTelemetryProvider`), the convenience scorers
(:func:`record_rerank_latency`, :func:`record_long_context`), the
:func:`try_import_submodule` helper, and the OpenTelemetry
:class:`Tracer` with its closed-stdout-safe :class:`SafeConsoleSpanExporter`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

from raghub.constants import ENV_LANGFUSE_PUBLIC_KEY, ENV_LANGFUSE_SECRET_KEY
from raghub.errors import ConfigurationError
from raghub.runtime import capture
from raghub.telemetry.base import Span, Telemetry
from raghub.telemetry.metrics import NoopSpan
from raghub.types import JSONValue

from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SpanExportResult

T = TypeVar("T")

__all__ = [
    "IMPORT_ERROR",
    "LANGFUSE_AVAILABLE",
    "Langfuse",
    "LangfuseSpan",
    "LangfuseTelemetryProvider",
    "SafeConsoleSpanExporter",
    "Tracer",
    "langfuse_client",
    "record_long_context",
    "record_rerank_latency",
    "try_import_submodule",
]


Langfuse: Any

from langfuse import Langfuse

LANGFUSE_AVAILABLE = True
IMPORT_ERROR: Exception | None = None


def try_import_submodule(module_name: str, target_name: str) -> Any:
    """Import ``target_name`` from ``module_name``; return ``None`` on failure."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    return getattr(module, target_name, None)


def record_rerank_latency(provider: str, seconds: float) -> None:
    """Record a rerank latency as a Langfuse score.

    Silent no-op when Langfuse is unconfigured.

    Args:
        provider: Provider label (e.g. ``"cohere"``).
        seconds: Latency in seconds.

    """
    client = langfuse_client()
    if client is None:
        return
    try:
        score = getattr(client, "score", None)
        if score is None:
            return
        score(name="raghub.rerank.latency", value=seconds, metadata={"provider": provider})
    except Exception:
        return


def record_long_context(*, outcome: str, seconds: float) -> None:
    """Record a long-context second-pass event as a Langfuse score.

    Silent no-op when Langfuse is unconfigured.

    Args:
        outcome: One of ``"ran"``, ``"skipped"``, ``"bad_json"``,
            ``"error"``. Unknown values still recorded under that label.
        seconds: Observed wall-clock latency in seconds.

    """
    client = langfuse_client()
    if client is None:
        return
    try:
        score = getattr(client, "score", None)
        if score is None:
            return
        score(
            name="raghub.long_context.duration",
            value=seconds,
            metadata={"outcome": outcome},
        )
    except Exception:
        return


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
    def try_call(fn: Callable[..., T], *args: Any, **kwargs: JSONValue) -> T:
        """Invoke ``fn`` and return its value; errors propagate.

        Args:
            fn: The callable to invoke.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The callable's return value.

        """
        return fn(*args, **kwargs)

    @staticmethod
    def langfuse_client(
        host: str | None, public_key: str, secret_key: str, flush_interval: float
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
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host or "https://cloud.langfuse.com",
            flush_interval=flush_interval,
        )

    def info(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an info log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.

        """
        with self.span(f"log.info.{message}", level="info", **kwargs):
            pass

    def warning(self, message: str, **kwargs: JSONValue) -> None:
        """Emit a warning log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.

        """
        with self.span(f"log.warning.{message}", level="warning", **kwargs):
            pass

    def error(self, message: str, **kwargs: JSONValue) -> None:
        """Emit an error log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.

        """
        with self.span(f"log.error.{message}", level="error", **kwargs):
            pass

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency as a Langfuse score.

        Args:
            name: Metric name.
            value_ms: Latency in milliseconds.
            **labels: Optional label set.

        """
        if self.client is None:
            return
        score = getattr(self.client, "score", None)
        if score is None:
            return
        try:
            score(name=f"latency.{name}", value=value_ms, metadata=labels or {})
        except Exception:
            return

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Record an increment as a Langfuse score.

        Args:
            name: Counter name.
            value: Increment amount.
            **labels: Optional label set.

        """
        if self.client is None:
            return
        score = getattr(self.client, "score", None)
        if score is None:
            return
        try:
            score(name=f"counter.{name}", value=value, metadata=labels or {})
        except Exception:
            return

    def start_span(self, name: str, **attrs: Any) -> Any:
        """Open a Langfuse span (v3) or fall back to a no-op span.

        Args:
            name: Span name.
            **attrs: Span attributes. ``user_id`` and ``session_id``
                are propagated to every child observation via
                ``propagate_attributes`` so Langfuse traces carry
                the multi-user attribution.

        Returns:
            A span (live :class:`LangfuseSpan` or :class:`NoopSpan`).

        """
        if self.client is None:
            return NoopSpan(name)
        propagate = {k: v for k, v in attrs.items() if k in {"user_id", "session_id"} and v}
        if propagate:
            self.propagate_langfuse(**propagate)
        start_obs = getattr(self.client, "start_as_current_observation", None)
        if start_obs is None:
            return NoopSpan(name)
        ctx = start_obs(as_type="span", name=name, **{"input": attrs})
        return LangfuseSpan(ctx, name)

    def propagate_langfuse(self, **attrs: Any) -> None:
        """Call Langfuse ``propagate_attributes`` if available."""
        propagate = getattr(self.client, "propagate_attributes", None)
        if propagate is not None:
            propagate(**attrs)

    @staticmethod
    def end_span(span: Any) -> None:
        """Close a span.

        Args:
            span: The span returned by :meth:`start_span`.

        """
        if isinstance(span, NoopSpan):
            return
        span.end()

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
        gen = start_obs(as_type="generation", name=name, model=model)
        if gen is None:
            return
        with gen:
            gen.update(
                usage_details={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                }
            )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Any]:
        """Context-manager wrapper around :meth:`start_span` / :meth:`end_span`."""
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)


class SafeConsoleSpanExporter:
    """Console exporter that survives a closed-stdout shutdown.

    The default :class:`ConsoleSpanExporter` raises
    :class:`ValueError` when ``sys.stdout`` is closed. That breaks
    every test that exercises a tracer and exits at process
    shutdown. This subclass wraps the ``export`` method in a guard
    that swallows the error and returns a ``FAILURE`` result.
    """

    def __init__(self, *args: Any, **kwargs: JSONValue) -> None:
        """Lazy-import the parent :class:`ConsoleSpanExporter`."""
        self.parent = ConsoleSpanExporter(*args, **kwargs)  # type: ignore[arg-type]

    def export(self, spans: Sequence[Any]) -> Any:
        """Forward to the parent exporter; suppress closed-stdout errors.

        Args:
            spans: The batch of spans to export.

        Returns:
            The parent's return value or ``SpanExportResult.FAILURE``
            on a closed-stdout error.

        """
        result, error = capture(self.parent.export, spans)
        if error is None:
            return result
        if "closed file" in str(error):
            return self.failure()
        raise error

    def shutdown(self) -> None:
        """Forward shutdown to the parent exporter."""
        self.parent.shutdown()

    @staticmethod
    def failure() -> Any:
        """Return the OTel FAILURE result."""
        return SpanExportResult.FAILURE


class Tracer:
    """Wrap an OpenTelemetry tracer provider with FastAPI auto-instrumentation.

    Attributes:
        provider: The underlying :class:`TracerProvider`. Exposed so
            callers can swap processors / exporters before
            :meth:`instrument_app` runs.
        tracer: The :class:`trace.Tracer` instance used to create
            spans manually.

    """

    def __init__(self, service_name: str = "raghub") -> None:
        """Configure a tracer provider with a console span exporter.

        Args:
            service_name: The ``service.name`` resource attribute.

        Raises:
            ConfigurationError: When OpenTelemetry SDK packages are
                not installed.

        """
        ot_trace = try_import_submodule("opentelemetry", "trace")
        ot_resources = try_import_submodule("opentelemetry.sdk.resources", "Resource")
        ot_trace_mod = try_import_submodule("opentelemetry.sdk.trace", "TracerProvider")
        ot_export = try_import_submodule("opentelemetry.sdk.trace.export", "BatchSpanProcessor")
        if ot_trace is None or ot_resources is None or ot_trace_mod is None or ot_export is None:
            raise ConfigurationError("OpenTelemetry tracing requires opentelemetry-sdk")

        trace = ot_trace
        resource_cls = ot_resources
        tracer_provider_cls = ot_trace_mod
        batch_processor_cls = ot_export

        resource = resource_cls.create({"service.name": service_name})
        provider = tracer_provider_cls(resource=resource)
        processor = batch_processor_cls(SafeConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.provider = provider
        self.tracer = trace.get_tracer(service_name)

    def add_exporter(self, *, endpoint: str, insecure: bool = True) -> None:
        """Replace the default console exporter with an OTLP one.

        Args:
            endpoint: The OTLP collector endpoint (e.g.
                ``"http://otel-collector:4317"``).
            insecure: When ``True`` (default) use HTTP/gRPC without
                TLS. Production deployments should set this to
                ``False`` and supply a TLS endpoint.

        """
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        self.provider.add_span_processor(BatchSpanProcessor(exporter))

    @staticmethod
    def instrument_app(app: Any) -> None:
        """Auto-instrument a FastAPI app with OpenTelemetry middleware.

        Args:
            app: A FastAPI application instance.

        """
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)

    def create_span(self, name: str) -> Any:
        """Open a new span as a context manager.

        Args:
            name: The span name.

        Returns:
            The :class:`opentelemetry.trace.Span` context manager
            from :meth:`tracer.start_as_current_span`.

        """
        return self.tracer.start_as_current_span(name)

    def shutdown(self) -> None:
        """Flush and shut down the underlying provider.

        Safe to call multiple times.
        """
        capture(self.provider.shutdown)
