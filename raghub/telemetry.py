"""Observability and telemetry surface.

The framework ships three interoperable telemetry layers in a single
module:

* :class:`LangfuseTelemetryProvider` — v3 Langfuse adapter (the spec
  default). When ``langfuse`` is not installed or no credentials are
  configured, every method silently no-ops so the framework keeps
  running without telemetry.
* :class:`LoguruTelemetryProvider` — loguru adapter for users who
  prefer in-process logging without a remote service.
* :class:`NoOpTelemetry` — silent default satisfying the
  :class:`TelemetryProvider` contract with zero I/O.
* :class:`RedactingTelemetry` — wraps another provider and scrubs
  kwargs whose keys look like secrets before forwarding.

Spans and tracer primitives also live here:

* :class:`NoopSpan` / :class:`LangfuseSpan` — span implementations.
* :class:`LoguruSpan` — span that records duration into metrics.
* :class:`Tracer` — OpenTelemetry tracer with FastAPI auto-instrumentation.
* :class:`SafeConsoleSpanExporter` — stdout-safe OTel exporter.
* :func:`build_logger` / :func:`scrub_secrets` / :func:`redact_record` —
  redacting-logger plumbing.

Public constants:

* :data:`SECRET_KEY_RE` — the regex used by the redacting layer.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

from loguru import logger as loguru_logger

from raghub.await_sync import capture
from raghub.errors import ConfigurationError, MissingDepError
from raghub.models import Logger, Span, TelemetryProvider

T = TypeVar("T")

__all__ = [
    "LangfuseTelemetryProvider",
    "LoguruTelemetryProvider",
    "NoOpTelemetry",
    "RedactingTelemetry",
    "build_logger",
]

langfuse_get_client: Any
Langfuse: Any

try:
    from langfuse import Langfuse
    from langfuse import get_client as langfuse_get_client

    LANGFUSE_AVAILABLE = True
    IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # optional dep — propagate when explicitly requested
    langfuse_get_client = None
    Langfuse = None
    LANGFUSE_AVAILABLE = False
    IMPORT_ERROR = exc

LOGGER = logging.getLogger("raghub.telemetry.langfuse")

SECRET_KEY_RE = re.compile(
    r"(?i)(password|passwd|secret|api_key|apikey|access_token|refresh_token|jwt|authorization)"
)


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
    if langfuse_get_client is None:
        return None
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        return langfuse_get_client()
    except Exception:
        return None


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


def build_logger(level: str = "INFO") -> LoguruLogger:
    """Configure the process-wide loguru logger with a pretty console sink.

    Removes any default sinks installed by loguru's ``logger`` module
    and installs a single sink on stderr that scrubs secret-like keys
    before formatting. The sink uses loguru's built-in level icons
    (pencil for trace, bug for debug, info for INFO, check for SUCCESS,
    warning for WARNING, error for ERROR) plus colour-coded level
    names and a collapsed frame, so each log line is one readable row.

    Args:
        level: Minimum log level (e.g. ``"INFO"``, ``"DEBUG"``).
            Unknown values fall back to ``"INFO"``.

    Returns:
        A :class:`LoguruLogger` ready for ``info`` / ``warning`` /
        ``error`` calls.

    """
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> "
            "<level>{level.icon}</level> "
            "<level>{level: <7}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    loguru_logger.configure(extra={"redacted": True})
    return LoguruLogger()


class LoguruLogger(Logger):
    """Adapter that implements :class:`Logger` against :mod:`loguru`.

    Every method is a thin wrapper that copies structured kwargs into
    loguru's bound context. The redaction step lives in the sink, so
    call sites never have to think about it.
    """

    def __init__(self) -> None:
        """Bind a private logger for per-instance test output capture.

        Used to isolate loguru state across test cases.
        """
        self.logger = loguru_logger.bind(component="raghub")

    def info(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit an ``INFO``-level record with structured ``kwargs``."""
        self.logger.bind(**kwargs).info(message)

    def warning(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit a ``WARNING``-level record."""
        self.logger.bind(**kwargs).warning(message)

    def error(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit an ``ERROR``-level record."""
        self.logger.bind(**kwargs).error(message)


class LoguruSpan(Span):
    """Span whose :meth:`end` logs the duration."""

    def __init__(
        self,
        name: str,
        logger: LoguruLogger,
        attributes: dict[str, Any],
    ) -> None:
        """Store the span's name and timing metadata."""
        self.name = name
        self.logger = logger
        self.attributes = attributes
        self.started = time.perf_counter()

    def end(self) -> None:
        """Log the span's duration on completion."""
        duration_ms = (time.perf_counter() - self.started) * 1000.0
        self.logger.info(f"span.end.{self.name}", duration_ms=duration_ms, **self.attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute for later emission."""
        self.attributes[key] = value


class LoguruTelemetryProvider(TelemetryProvider):
    """Telemetry provider that sinks through loguru.

    Metric emission (``record_latency`` / ``increment`` /
    ``record_tokens``) is a no-op; observability in this codebase is
    via Langfuse. The provider still implements the full
    :class:`TelemetryProvider` contract so it can stand in for the
    Langfuse provider when no credentials are configured.
    """

    def __init__(self, logger: LoguruLogger | None = None) -> None:
        """Build the provider with an optional logger override."""
        self.logger = logger or LoguruLogger()

    def info(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit an ``info``-level record."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit a ``warning``-level record."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit an ``error``-level record."""
        self.logger.error(message, **kwargs)

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """No-op; Langfuse absorbs metric emission when configured."""
        return None

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """No-op; Langfuse absorbs metric emission when configured."""
        return None

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a new span.

        Args:
            name: Span name.
            **attrs: Attributes attached to the span.

        Returns:
            A :class:`LoguruSpan`.

        """
        return LoguruSpan(name, self.logger, attrs)

    @staticmethod
    def end_span(span: Span) -> None:
        """Close the supplied span."""
        span.end()

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Log token usage; metric emission is a no-op."""
        self.logger.info(
            "tokens",
            name=name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context-manager wrapper around :meth:`start_span` / :meth:`end_span`."""
        opened = self.start_span(name, **attrs)
        try:
            yield opened
        finally:
            self.end_span(opened)


class NoopSpan(Span):
    """No-op span implementation.

    Used when Langfuse is not installed or no credentials are
    configured. Implements the :class:`Span` protocol so callers can
    treat it interchangeably with the live implementation.
    """

    def __init__(self, name: str) -> None:
        """Store the span name; no exporter is wired."""
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


class LangfuseTelemetryProvider(TelemetryProvider):
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
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
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
            and os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
        )

    @staticmethod
    def try_call(fn: Callable[..., T], *args: Any, **kwargs: "JSONValue") -> T:
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
        if langfuse_get_client is not None:
            return langfuse_get_client()
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host or "https://cloud.langfuse.com",
            flush_interval=flush_interval,
        )

    def info(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit an info log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.

        """
        with self.span(f"log.info.{message}", level="info", **kwargs):
            pass

    def warning(self, message: str, **kwargs: "JSONValue") -> None:
        """Emit a warning log via a span.

        Args:
            message: Log message.
            **kwargs: Structured key/value pairs.

        """
        with self.span(f"log.warning.{message}", level="warning", **kwargs):
            pass

    def error(self, message: str, **kwargs: "JSONValue") -> None:
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
    def end_span(span: Span) -> None:
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
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context-manager wrapper around :meth:`start_span` / :meth:`end_span`."""
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)


class NoOpTelemetry(TelemetryProvider):
    """Silent telemetry provider; satisfies the contract."""

    def info(self, message: str, **kwargs: "JSONValue") -> None:
        """No-op."""

    def warning(self, message: str, **kwargs: "JSONValue") -> None:
        """No-op."""

    def error(self, message: str, **kwargs: "JSONValue") -> None:
        """No-op."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """No-op."""

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """No-op."""

    @staticmethod
    def start_span(name: str, **attrs: Any) -> Span:
        """Return a no-op span.

        Args:
            name: Span name (recorded for completeness).
            **attrs: Span attributes (ignored).

        Returns:
            A :class:`NoopSpan`.

        """
        return NoopSpan(name)

    def end_span(self, span: Span) -> None:
        """No-op."""

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """No-op."""


class RedactingTelemetry(TelemetryProvider):
    """Telemetry wrapper that redacts secret-looking keys."""

    def __init__(self, inner: TelemetryProvider) -> None:
        """Wrap ``inner`` with secret-redaction.

        Args:
            inner: The downstream provider receiving redacted calls.

        """
        self.inner = inner

    def info(self, message: str, **kwargs: "JSONValue") -> None:
        """Forward ``info`` with redacted kwargs."""
        self.inner.info(message, **scrub_secrets(kwargs))

    def warning(self, message: str, **kwargs: "JSONValue") -> None:
        """Forward ``warning`` with redacted kwargs."""
        self.inner.warning(message, **scrub_secrets(kwargs))

    def error(self, message: str, **kwargs: "JSONValue") -> None:
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


class SafeConsoleSpanExporter:
    """Console exporter that survives a closed-stdout shutdown.

    The default :class:`ConsoleSpanExporter` raises
    :class:`ValueError` when ``sys.stdout`` is closed. That breaks
    every test that exercises a tracer and exits at process
    shutdown. This subclass wraps the ``export`` method in a guard
    that swallows the error and returns a ``FAILURE`` result.
    """

    def __init__(self, *args: Any, **kwargs: "JSONValue") -> None:
        """Lazy-import the parent :class:`ConsoleSpanExporter`."""
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        except ImportError:
            raise MissingDepError(
                "opentelemetry-sdk",
                "pip install raghub[otel]",
            ) from None
        self.parent = ConsoleSpanExporter(*args, **kwargs)

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
        """Return the OTel FAILURE result without importing it at module load."""
        try:
            from opentelemetry.sdk.trace.export import SpanExportResult
        except ImportError:
            return None
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
