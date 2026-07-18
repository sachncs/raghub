"""Observability tests covering the loguru-backed logger, Prometheus
metrics, and the OpenTelemetry tracer.

These tests verify the public surface only; they don't try to
introspect log output or scrape the Prometheus exposition format.
"""

from __future__ import annotations

from raghub.observability.logging import LoguruLogger, build_logger
from raghub.observability.metrics import NullMetrics, PrometheusMetrics
from raghub.observability.tracing import Tracer


class TestBuildLogger:
    def test_returns_loguru_logger(self):
        """``build_logger`` returns a configured :class:`LoguruLogger`."""
        logger = build_logger("INFO")
        assert isinstance(logger, LoguruLogger)
        logger.info("test message")
        logger.warning("test warning")
        logger.error("test error")


class TestLoguruLogger:
    def test_info_warning_error_delegate(self):
        """Logger methods delegate to the underlying sink without raising."""
        logger = build_logger("INFO")
        wrapped = logger
        wrapped.info("hello", request_id="abc")
        wrapped.warning("careful", request_id="abc")
        wrapped.error("boom", request_id="abc")


class TestNullMetrics:
    def test_noops(self):
        """NullMetrics silently accepts every method."""
        metrics = NullMetrics()
        metrics.record_latency("query", 1.0)
        metrics.record_latency("ingestion", 2.0)
        metrics.increment("auth_total", 1)
        metrics.increment("error_total", 1)


class TestPrometheusMetrics:
    def test_initializes(self):
        """PrometheusMetrics instantiates without raising."""
        metrics = PrometheusMetrics()
        assert metrics is not None

    def test_record_query(self):
        """``record_query`` is idempotent and accepts floats."""
        metrics = PrometheusMetrics()
        metrics.record_query(100.0, 5)
        metrics.record_query(200.0, 3)

    def test_record_ingestion(self):
        """``record_ingestion`` records duration and chunk count."""
        metrics = PrometheusMetrics()
        metrics.record_ingestion(500.0, 20)

    def test_record_auth(self):
        """``record_auth`` observes duration and success label."""
        metrics = PrometheusMetrics()
        metrics.record_auth(10.0, True)
        metrics.record_auth(15.0, False)

    def test_record_error(self):
        """``record_error`` increments the error counter."""
        metrics = PrometheusMetrics()
        metrics.record_error("auth_error")

    def test_routes_latency_to_matching_histograms(self):
        """Latency observation names route to the right histograms."""
        from prometheus_client import REGISTRY

        metrics = PrometheusMetrics()
        ingestion_before = REGISTRY.get_sample_value("raghub_ingestion_duration_ms_count") or 0
        auth_before = REGISTRY.get_sample_value("raghub_auth_duration_ms_count") or 0
        query_before = REGISTRY.get_sample_value("raghub_query_duration_ms_count") or 0

        metrics.record_latency("span.ingest.upsert", 10)
        metrics.record_latency("span.auth.login", 20)

        assert (
            REGISTRY.get_sample_value("raghub_ingestion_duration_ms_count") == ingestion_before + 1
        )
        assert REGISTRY.get_sample_value("raghub_auth_duration_ms_count") == auth_before + 1
        assert REGISTRY.get_sample_value("raghub_query_duration_ms_count") == query_before

    def test_routes_tokens_to_public_counters(self):
        """Token counters route to the public ``raghub_*_tokens_total`` series."""
        from prometheus_client import REGISTRY

        metrics = PrometheusMetrics()
        labels = {"model": "routing-test"}
        prompt_before = REGISTRY.get_sample_value("raghub_prompt_tokens_total", labels) or 0
        completion_before = REGISTRY.get_sample_value("raghub_completion_tokens_total", labels) or 0

        metrics.increment("tokens.prompt", 7, **labels)
        metrics.increment("tokens.completion", 11, **labels)

        assert REGISTRY.get_sample_value("raghub_prompt_tokens_total", labels) == prompt_before + 7
        assert (
            REGISTRY.get_sample_value("raghub_completion_tokens_total", labels)
            == completion_before + 11
        )

    def test_register_app(self):
        """``register_app`` exposes ``/metrics`` on a FastAPI app."""
        from fastapi import FastAPI

        metrics = PrometheusMetrics()
        app = FastAPI()
        metrics.register_app(app)
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/metrics" in paths


class TestTracer:
    def test_instantiation(self):
        """Constructing :class:`Tracer` is safe when the SDK is present."""
        tracer = Tracer("raghub-test")
        assert tracer is not None
        tracer.shutdown()
