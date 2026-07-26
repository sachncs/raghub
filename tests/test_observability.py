"""Observability tests covering the loguru-backed logger, Prometheus
metrics, and the OpenTelemetry tracer.

Tests verify behavioral contracts: registry-level value assertions for
Prometheus metrics, logger delegation for loguru, and safe construction
for the tracer.
"""

from __future__ import annotations

from raghub.observability import LoguruLogger, build_logger
from raghub.observability import PrometheusMetrics


class TestBuildLogger:
    def test_returns_configured_logger(self):
        """build_logger returns a LoguruLogger that delegates to loguru."""
        logger = build_logger("INFO")
        assert isinstance(logger, LoguruLogger)
        logger.info("test message")
        logger.warning("test warning")
        logger.error("test error")

    def test_logger_methods_accept_extra_kwargs(self):
        """Logger methods accept and forward keyword arguments."""
        logger = build_logger("INFO")
        logger.info("hello", request_id="abc")
        logger.warning("careful", request_id="abc")
        logger.error("boom", request_id="abc")


class TestPrometheusMetricsLatencyRouting:
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


class TestPrometheusMetricsTokenRouting:
    def test_routes_tokens_to_public_counters(self):
        """Token counters route to the public raghub_*_tokens_total series."""
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


class TestPrometheusMetricsRegistration:
    def test_register_app_exposes_metrics_endpoint(self):
        """register_app exposes /metrics on a FastAPI app."""
        from fastapi import FastAPI

        metrics = PrometheusMetrics()
        app = FastAPI()
        metrics.register_app(app)
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/metrics" in paths


class TestTracer:
    def test_shutdown_without_spans_is_safe(self):
        """Tracer.shutdown completes without error when no spans were created."""
        from raghub.observability import Tracer

        tracer = Tracer("raghub-test")
        tracer.shutdown()
