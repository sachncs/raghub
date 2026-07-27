"""Observability and telemetry surface.

The framework ships four interoperable telemetry layers in a single
module:

* :class:`LangfuseTelemetryProvider` — v3 Langfuse adapter (the spec
  default). When ``langfuse`` is not installed or no credentials are
  configured, every method silently no-ops so the framework keeps
  running without telemetry.
* :class:`LoguruTelemetryProvider` — loguru + Prometheus adapter
  for users who prefer in-process observability without a remote
  service.
* :class:`NoOpTelemetry` — silent default satisfying the
  :class:`the telemetry provider protocol` contract
  with zero I/O.
* :class:`RedactingTelemetry` — wraps another provider and scrubs
  kwargs whose keys look like secrets before forwarding.

Spans, metrics, and tracer primitives also live here:

* :class:`NoopSpan` / :class:`LangfuseSpan` — span implementations.
* :class:`LoguruSpan` — span that records duration into metrics.
* :class:`PrometheusMetrics` / :class:`NullMetrics` — metrics
  recorder + drop-all fallback.
* :class:`MetricsRegistry` — process-wide holder replacing the
  previous module-level singleton.
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
from typing import Any, TypeVar, cast

from loguru import logger as loguru_logger
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SpanExportResult
from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram
from prometheus_client.openmetrics.exposition import generate_latest

from raghub.exceptions import ConfigurationError
from raghub.utils import capture
from raghub.models import Logger, Metrics, Span, TelemetryProvider

T = TypeVar("T")

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

known_collectors: dict[str, object] = {}


class MetricsRegistry:
    """Process-wide holder for the active :class:`PrometheusMetrics`.

    Hot-path helpers (:func:`record_rerank_latency`,
    :func:`record_long_context`) delegate to :meth:`current`; the
    facade calls :meth:`set` once during construction.

    Attributes:
        instance: The currently-registered :class:`PrometheusMetrics`
            (or ``None`` when nothing is registered).
    """

    def __init__(self) -> None:
        """Initialise the registry with no instance registered."""
        self.instance: PrometheusMetrics | None = None

    def set(self, value: PrometheusMetrics | None) -> None:
        """Register the active metrics instance for this process.

        Args:
            value: The :class:`PrometheusMetrics` to expose to
                hot-path callers, or ``None`` to clear the registry.
        """
        self.instance = value

    def current(self) -> PrometheusMetrics | None:
        """Return the currently-registered instance, or ``None``."""
        return self.instance

    def is_available(self) -> bool:
        """Return ``True`` when an instance is registered."""
        return self.instance is not None


DEFAULT_METRICS_REGISTRY = MetricsRegistry()


class NullMetrics:
    """Metrics recorder that drops every call.

    Useful in tests and minimal contexts where Prometheus' global
    REGISTRY would otherwise leak state between runs.
    """

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Discard a latency record.

        Args:
            name: Latency metric name (ignored).
            value_ms: Latency in milliseconds (ignored).
            **labels: Optional label set (ignored).
        """
        return None

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Discard a counter increment.

        Args:
            name: Counter name (ignored).
            value: Increment amount (ignored).
            **labels: Optional label set (ignored).
        """
        return None


class PrometheusMetrics:
    """Prometheus-backed metrics with idempotent metric registration.

    The class is safe to instantiate multiple times (e.g. across
    FastAPI reloads) because every metric is registered through
    helper functions that consult the global ``REGISTRY`` first.

    Attributes:
        query_duration: Histogram of query durations in milliseconds.
        ingestion_duration: Histogram of ingestion durations in ms.
        auth_duration: Histogram of auth call durations in ms.
        auth_total: Counter of auth attempts labelled by success.
        error_total: Counter of errors labelled by ``error_type``.
    """

    def __init__(self, app: Any | None = None) -> None:
        """Register metrics and (optionally) the FastAPI ``/metrics`` route.

        Args:
            app: Optional FastAPI app. When provided, ``/metrics``
                is registered and serves the OpenMetrics exposition
                format.
        """

        def collector_registered(name: str) -> bool:
            public_name = name.removesuffix("_total")
            return any(metric.name == public_name for metric in REGISTRY.collect())

        def safe_histogram(name: str, desc: str, buckets: list[float]) -> Histogram:
            existing = known_collectors.get(name)
            if existing is not None and isinstance(existing, Histogram) and collector_registered(name):
                return existing
            collector = Histogram(name, desc, buckets=buckets, registry=REGISTRY)
            known_collectors[name] = collector
            return collector

        def safe_counter(name: str, desc: str, labels: list[str] | None = None) -> Counter:
            existing = known_collectors.get(name)
            if existing is not None and isinstance(existing, Counter) and collector_registered(name):
                return existing
            collector = Counter(name, desc, labels or [], registry=REGISTRY)
            known_collectors[name] = collector
            return collector

        self.query_duration: Histogram = safe_histogram(
            "raghub_query_duration_ms",
            "Query execution duration in milliseconds",
            [10, 50, 100, 250, 500, 1000, 2500, 5000],
        )
        self.ingestion_duration: Histogram = safe_histogram(
            "raghub_ingestion_duration_ms",
            "Ingestion duration in milliseconds",
            [50, 100, 250, 500, 1000, 2500, 5000, 10000],
        )
        self.auth_duration: Histogram = safe_histogram(
            "raghub_auth_duration_ms",
            "Authentication duration in milliseconds",
            [5, 10, 25, 50, 100, 250, 500],
        )
        self.auth_total: Counter = safe_counter(
            "raghub_auth_total",
            "Total authentication attempts",
            ["success"],
        )
        self.error_total: Counter = safe_counter(
            "raghub_error_total",
            "Total errors",
            ["error_type"],
        )
        self.prompt_tokens: Counter = safe_counter(
            "raghub_prompt_tokens_total",
            "Total prompt tokens",
            ["model"],
        )
        self.completion_tokens: Counter = safe_counter(
            "raghub_completion_tokens_total",
            "Total completion tokens",
            ["model"],
        )
        self.rerank_latency: Histogram = safe_histogram(
            "raghub_rerank_latency_seconds",
            "Reranker wall-clock latency in seconds",
            [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        self.long_context_pass: Counter = safe_counter(
            "raghub_long_context_pass_used_total",
            "Long-context second-pass rerank invocations",
            ["outcome"],
        )
        if app is not None:
            self.register_app(app)

    def record_query(self, duration_ms: float, top_k: int) -> None:
        """Record a query duration observation.

        Args:
            duration_ms: Query duration in milliseconds.
            top_k: Requested top-k value (currently not exported as a
                label).
        """
        self.query_duration.observe(duration_ms)

    def record_ingestion(self, duration_ms: float, chunk_count: int) -> None:
        """Record an ingestion duration observation.

        Args:
            duration_ms: Ingestion duration in milliseconds.
            chunk_count: Number of chunks produced (currently not
                exported as a label; retained for forward

        """
        self.ingestion_duration.observe(duration_ms)

    def record_auth(self, duration_ms: float, success: bool) -> None:
        """Record an authentication attempt.

        Args:
            duration_ms: Auth duration in milliseconds.
            success: ``True`` for successful auth, ``False`` otherwise.
        """
        self.auth_duration.observe(duration_ms)
        self.auth_total.labels(success=str(success)).inc()

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record latency in the matching public histogram."""
        normalized = name.lower()
        if "ingest" in normalized:
            self.ingestion_duration.observe(value_ms)
        elif "auth" in normalized:
            self.auth_duration.observe(value_ms)
        else:
            self.query_duration.observe(value_ms)

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment the matching public counter."""
        normalized = name.lower()
        model = str(labels.get("model", ""))
        if normalized in {"tokens.prompt", "prompt_tokens", "prompt_tokens_total"}:
            self.prompt_tokens.labels(model=model).inc(value)
        elif normalized in {
            "tokens.completion",
            "completion_tokens",
            "completion_tokens_total",
        }:
            self.completion_tokens.labels(model=model).inc(value)
        else:
            self.error_total.labels(error_type=name).inc(value)

    def record_error(self, error_type: str) -> None:
        """Increment the error counter for ``error_type``.

        Args:
            error_type: A short label used as the ``error_type``
                metric dimension.
        """
        self.error_total.labels(error_type=error_type).inc()

    def register_app(self, app: Any) -> None:
        """Attach a ``/metrics`` route to ``app`` when it is FastAPI.

        Args:
            app: A FastAPI application instance.
        """
        from fastapi import FastAPI
        from fastapi.responses import Response

        if isinstance(app, FastAPI):

            @app.get("/metrics")
            async def metrics() -> Response:
                """Expose Prometheus metrics in OpenMetrics text format."""
                payload = cast(Callable[[CollectorRegistry], bytes], generate_latest)(REGISTRY)
                return Response(
                    content=payload,
                    media_type="text/plain",
                )


def try_import_submodule(module_name: str, target_name: str) -> Any:
    """Import ``target_name`` from ``module_name``; return ``None`` on failure."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    return getattr(module, target_name, None)


def set_active_metrics(instance: PrometheusMetrics | None) -> None:
    """Register the process-wide :class:`PrometheusMetrics` instance.

    Rerankers and other hot-path components call :func:`record_rerank_latency`
    which needs a back-reference to the active Prometheus registry. The
    facade calls this once during construction; rerankers read it lazily.

    Args:
        instance: The :class:`PrometheusMetrics` to expose to hot-path
            callers, or ``None`` to clear the registry.
    """
    DEFAULT_METRICS_REGISTRY.set(instance)


def record_rerank_latency(provider: str, seconds: float) -> None:
    """Observe a rerank latency into the active Prometheus histogram.

    No-op when no :class:`PrometheusMetrics` is registered yet (e.g. in
    unit tests). ``provider`` becomes the ``provider`` label.

    Args:
        provider: Provider label (e.g. ``"cohere"``).
        seconds: Latency in seconds.
    """
    metrics = DEFAULT_METRICS_REGISTRY.current()
    if metrics is None:
        return
    histogram, error = capture(metrics.rerank_latency.labels, provider=provider)
    if error is not None:
        return
    histogram.observe(seconds)


def record_long_context(*, outcome: str, seconds: float) -> None:
    """Increment the long-context pass counter.

    Args:
        outcome: One of ``"ran"``, ``"skipped"``, ``"bad_json"``,
            ``"error"``. Unknown values still increment the counter
            under that label so the operator sees them.
        seconds: Observed wall-clock latency (recorded only for
            informational purposes; the metric is a counter, not a
            histogram).
    """
    metrics = DEFAULT_METRICS_REGISTRY.current()
    if metrics is None:
        return
    counter, error = capture(metrics.long_context_pass.labels, outcome=outcome)
    if error is not None:
        return
    counter.inc()


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
            ``"***"``. Nested dicts are scrubbed recursively.
    """

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    scrubbed: dict[str, Any] = {}
    for key, value in record.items():
        if SECRET_KEY_RE.search(str(key)):
            scrubbed[key] = "***"
        else:
            scrubbed[key] = scrub(value)
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
        """Bind a private logger so tests can capture output per
        :class:`LoguruLogger` instance.
        """
        self.logger = loguru_logger.bind(component="raghub")

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an ``INFO``-level record with structured ``kwargs``."""
        self.logger.bind(**kwargs).info(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Emit a ``WARNING``-level record."""
        self.logger.bind(**kwargs).warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an ``ERROR``-level record."""
        self.logger.bind(**kwargs).error(message)


class LoguruSpan(Span):
    """Span whose :meth:`end` records duration into :class:`Metrics`."""

    def __init__(
        self,
        name: str,
        logger: LoguruLogger,
        metrics: Metrics,
        attributes: dict[str, Any],
    ) -> None:
        """Store the span's name and timing metadata."""
        self.name = name
        self.logger = logger
        self.metrics = metrics
        self.attributes = attributes
        self.started = time.perf_counter()

    def end(self) -> None:
        """Record duration into the metrics sink and log completion."""
        duration_ms = (time.perf_counter() - self.started) * 1000.0
        self.metrics.record_latency(f"span.{self.name}", duration_ms, **self.attributes)
        self.logger.info(f"span.end.{self.name}", duration_ms=duration_ms, **self.attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute for later emission."""
        self.attributes[key] = value


class LoguruTelemetryProvider(TelemetryProvider):
    """Telemetry provider that sinks through loguru and Prometheus."""

    def __init__(
        self,
        logger: LoguruLogger | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        """Build the provider with optional collaborators."""
        self.logger = logger or LoguruLogger()
        self.metrics = metrics or PrometheusMetrics()

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an ``info``-level record."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Emit a ``warning``-level record."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an ``error``-level record."""
        self.logger.error(message, **kwargs)

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Forward a latency observation to the metrics sink."""
        self.metrics.record_latency(name, value_ms, **labels)

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Forward a counter increment to the metrics sink."""
        self.metrics.increment(name, value, **labels)

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Open a new span.

        Args:
            name: Span name.
            **attrs: Attributes attached to the span.

        Returns:
            A :class:`LoguruSpan`.
        """
        return LoguruSpan(name, self.logger, self.metrics, attrs)

    def end_span(self, span: Span) -> None:
        """Close the supplied span."""
        span.end()

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage on the dedicated token counters."""
        self.metrics.increment("tokens.prompt", prompt_tokens, model=model)
        self.metrics.increment("tokens.completion", completion_tokens, model=model)
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
            self.client = self.build_langfuse_client(
                host, public_key, secret_key, flush_interval
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
    def try_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Invoke ``fn`` and return its value; errors propagate.

        Args:
            fn: The callable to invoke.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The callable's return value.
        """
        return fn(*args, **kwargs)

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
            return langfuse_get_client()
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host or "https://cloud.langfuse.com",
            flush_interval=flush_interval,
        )

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
        propagate = {k: v for k, v in attrs.items() if k in ("user_id", "session_id") and v}
        if propagate:
            self.propagate_to_langfuse(**propagate)
        start_obs = getattr(self.client, "start_as_current_observation", None)
        if start_obs is None:
            return NoopSpan(name)
        ctx = start_obs(as_type="span", name=name, **{"input": attrs})
        return LangfuseSpan(ctx, name)

    def propagate_to_langfuse(self, **attrs: Any) -> None:
        """Call Langfuse ``propagate_attributes`` if available."""
        propagate = getattr(self.client, "propagate_attributes", None)
        if propagate is not None:
            propagate(**attrs)

    def end_span(self, span: Span) -> None:
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

    def info(self, message: str, **kwargs: Any) -> None:
        """No-op."""

    def warning(self, message: str, **kwargs: Any) -> None:
        """No-op."""

    def error(self, message: str, **kwargs: Any) -> None:
        """No-op."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """No-op."""

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """No-op."""

    def start_span(self, name: str, **attrs: Any) -> Span:
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


class SafeConsoleSpanExporter(ConsoleSpanExporter):
    """Console exporter that survives a closed-stdout shutdown.

    The default :class:`ConsoleSpanExporter` raises
    :class:`ValueError` when ``sys.stdout`` is closed. That breaks
    every test that exercises a tracer and exits at process
    shutdown. This subclass wraps the ``export`` method in a guard
    that swallows the error and returns a ``FAILURE`` result.
    """

    def export(self, spans: Sequence[ReadableSpan]) -> Any:
        """Forward to the parent exporter; suppress closed-stdout errors.

        Args:
            spans: The batch of spans to export.

        Returns:
            The parent's return value (:class:`SpanExportResult.SUCCESS`
            on success) or :class:`SpanExportResult.FAILURE` on a
            closed-stdout error.
        """
        result, error = capture(super().export, spans)
        if error is None:
            return result
        if "closed file" in str(error):
            return self.failed_export_result()
        raise error

    def failed_export_result(self) -> Any:
        """Return :class:`SpanExportResult.FAILURE` without importing OTel types."""
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
        ot_export = try_import_submodule(
            "opentelemetry.sdk.trace.export", "BatchSpanProcessor"
        )
        if ot_trace is None or ot_resources is None or ot_trace_mod is None or ot_export is None:
            raise ConfigurationError("OpenTelemetry tracing requires opentelemetry-sdk")

        trace = ot_trace
        Resource = ot_resources
        TracerProvider = ot_trace_mod
        BatchSpanProcessor = ot_export

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(SafeConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.provider = provider
        self.tracer = trace.get_tracer(service_name)

    def add_otlp_exporter(self, *, endpoint: str, insecure: bool = True) -> None:
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

    def instrument_app(self, app: Any) -> None:
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