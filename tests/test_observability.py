"""Observability tests covering the structured logger, Prometheus
metrics, and the OpenTelemetry tracer.

These tests verify the public surface only; they don't try to
introspect log output or scrape the Prometheus exposition format.
"""

from __future__ import annotations


from raghub.observability.logging import build_logger, StructuredLogger
from raghub.observability.metrics import NullMetrics, PrometheusMetrics
from raghub.observability.tracing import OpenTelemetryTracer


class TestBuildLogger:
    def test_returns_structlog_logger(self):
        logger = build_logger("INFO")
        assert logger is not None
        logger.info("test message")
        logger.warning("test warning")
        logger.error("test error")


class TestStructuredLogger:
    def test_wraps_logger(self):
        logger = build_logger("INFO")
        wrapped = StructuredLogger(logger)
        wrapped.info("hello")
        wrapped.warning("careful")
        wrapped.error("boom")


class TestNullMetrics:
    def test_noops(self):
        metrics = NullMetrics()
        metrics.record_latency("query", 1.0)
        metrics.record_latency("ingestion", 2.0)
        metrics.increment("auth_total", 1)
        metrics.increment("error_total", 1)


class TestPrometheusMetrics:
    def test_initializes(self):
        metrics = PrometheusMetrics()
        assert metrics is not None

    def test_record_query(self):
        metrics = PrometheusMetrics()
        metrics.record_query(100.0, 5)
        metrics.record_query(200.0, 3)

    def test_record_ingestion(self):
        metrics = PrometheusMetrics()
        metrics.record_ingestion(500.0, 20)

    def test_record_auth(self):
        metrics = PrometheusMetrics()
        metrics.record_auth(10.0, True)
        metrics.record_auth(15.0, False)

    def test_record_error(self):
        metrics = PrometheusMetrics()
        metrics.record_error("auth_error")


class TestOpenTelemetryTracer:
    def test_initializes(self):
        tracer = OpenTelemetryTracer("test-raghub")
        assert tracer is not None

    def test_create_span(self):
        tracer = OpenTelemetryTracer("test-raghub")
        with tracer.create_span("test-span"):
            pass
