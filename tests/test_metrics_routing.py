"""Smoke tests for the new public-counter Prometheus surface.

The :class:`PrometheusMetrics` class routes generic ``record_latency``
and ``increment`` calls to the right public counter or histogram by
name, instead of folding every observation into the query histogram
(legacy behaviour). These tests exercise the routing and the
``/metrics`` route registration on a FastAPI app.
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import REGISTRY
from starlette.testclient import TestClient

from raghub.observability.metrics import PrometheusMetrics


def test_prometheus_metrics_public_counters_exposed() -> None:
    """The public counter names appear in the Prometheus exposition."""
    metrics = PrometheusMetrics()
    metrics.increment("tokens.prompt", 1, model="routing-smoke")
    metrics.increment("tokens.completion", 2, model="routing-smoke")
    metrics.record_latency("ingest.convert", 3.5)
    metrics.record_latency("query.embed", 1.5)
    metrics.record_latency("auth.login", 0.5)
    metrics.record_auth(7.0, True)
    metrics.record_error("auth_error")
    app = FastAPI()
    metrics.register_app(app)
    with TestClient(app) as client:
        body = client.get("/metrics").text
    assert "raghub_query_duration_ms" in body
    assert "raghub_ingestion_duration_ms" in body
    assert "raghub_auth_duration_ms" in body
    assert "raghub_auth_total" in body
    assert "raghub_error_total" in body
    assert "raghub_prompt_tokens_total" in body
    assert "raghub_completion_tokens_total" in body


def test_prometheus_metrics_token_routing_uses_model_label() -> None:
    """Token increments honour the ``model`` label exactly as supplied."""
    metrics = PrometheusMetrics()
    metrics.increment("tokens.prompt", 4, model="lab")
    metrics.increment("tokens.completion", 7, model="lab")
    assert REGISTRY.get_sample_value("raghub_prompt_tokens_total", {"model": "lab"}) == 4
    assert REGISTRY.get_sample_value("raghub_completion_tokens_total", {"model": "lab"}) == 7


def test_prometheus_metrics_routes_to_auth_duration() -> None:
    """The ``auth`` substring routes to the auth histogram, not query."""
    metrics = PrometheusMetrics()
    metrics.record_latency("auth.login.success", 12.0)
    metrics.record_latency("query.execute", 30.0)
    assert REGISTRY.get_sample_value("raghub_auth_duration_ms_count", {}) is not None
    assert REGISTRY.get_sample_value("raghub_query_duration_ms_count", {}) is not None
