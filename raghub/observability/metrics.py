"""Prometheus metrics with a no-op fallback.

The :class:`PrometheusMetrics` class registers histograms for query,
ingestion, and authentication latencies plus counters for auth
attempts, errors, and LLM tokens. Callers that want the metrics surface but don't
want Prometheus client-side effects can use :class:`NullMetrics`
which silently drops every call.
"""

from __future__ import annotations

from typing import Any, cast

from prometheus_client import REGISTRY, Counter, Histogram
from prometheus_client.openmetrics.exposition import generate_latest

metric_collectors: dict[str, Counter | Histogram] = {}


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
            existing: Any = metric_collectors.get(name)
            if existing is not None and collector_registered(name):
                return cast(Histogram, existing)
            collector = Histogram(name, desc, buckets=buckets, registry=REGISTRY)
            metric_collectors[name] = collector
            return collector

        def safe_counter(name: str, desc: str, labels: list[str] | None = None) -> Counter:
            existing: Any = metric_collectors.get(name)
            if existing is not None and collector_registered(name):
                return cast(Counter, existing)
            collector = Counter(name, desc, labels or [], registry=REGISTRY)
            metric_collectors[name] = collector
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
        if app is not None:
            self.register_app(app)

    def record_query(self, duration_ms: float, top_k: int) -> None:
        """Record a query duration observation.

        Args:
            duration_ms: Query duration in milliseconds.
            top_k: Requested top-k value (currently not exported as a
                label; retained for forward compatibility).
        """
        self.query_duration.observe(duration_ms)

    def record_ingestion(self, duration_ms: float, chunk_count: int) -> None:
        """Record an ingestion duration observation.

        Args:
            duration_ms: Ingestion duration in milliseconds.
            chunk_count: Number of chunks produced (currently not
                exported as a label; retained for forward
                compatibility).
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
                return Response(
                    content=generate_latest(REGISTRY),
                    media_type="text/plain",
                )
