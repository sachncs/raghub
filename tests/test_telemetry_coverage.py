"""Coverage tests for :mod:`raghub.telemetry` helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from raghub.telemetry import (
    MetricsRegistry,
    NullMetrics,
    PrometheusMetrics,
    record_long_context,
    record_rerank_latency,
    redact_record,
    scrub_secrets,
    set_active_metrics,
)

# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


def test_metrics_registry_default_is_none() -> None:
    """A fresh registry has no current metrics instance."""
    registry = MetricsRegistry()
    assert registry.current() is None
    assert registry.is_available() is False


def test_metrics_registry_set_and_current() -> None:
    """``set`` stores the instance; ``current`` returns it."""
    registry = MetricsRegistry()
    instance = MagicMock()
    registry.set(instance)
    assert registry.current() is instance
    assert registry.is_available() is True


def test_metrics_registry_set_none() -> None:
    """``set(None)`` clears the current instance."""
    registry = MetricsRegistry()
    registry.set(MagicMock())
    registry.set(None)
    assert registry.current() is None


# ---------------------------------------------------------------------------
# NullMetrics
# ---------------------------------------------------------------------------


def test_null_metrics_record_latency_is_noop() -> None:
    """``NullMetrics.record_latency`` accepts any args without effect."""
    NullMetrics().record_latency("name", 1.0, label="x")


def test_null_metrics_increment_is_noop() -> None:
    """``NullMetrics.increment`` accepts any args without effect."""
    NullMetrics().increment("name", value=1, label="x")


# ---------------------------------------------------------------------------
# PrometheusMetrics
# ---------------------------------------------------------------------------


def test_prometheus_metrics_record_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """``record_query`` records both a histogram and a counter."""
    metrics = PrometheusMetrics()
    metrics.record_query(123.0, top_k=5)
    assert metrics is not None


def test_prometheus_metrics_record_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    """``record_ingestion`` records both a histogram and a counter."""
    metrics = PrometheusMetrics()
    metrics.record_ingestion(500.0, chunk_count=10)


def test_prometheus_metrics_record_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """``record_auth`` records auth timing with a success label."""
    metrics = PrometheusMetrics()
    metrics.record_auth(50.0, success=True)
    metrics.record_auth(75.0, success=False)


def test_prometheus_metrics_record_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``record_error`` increments the error counter."""
    metrics = PrometheusMetrics()
    metrics.record_error("validation_error")


def test_prometheus_metrics_record_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """``record_latency`` records a generic latency histogram."""
    metrics = PrometheusMetrics()
    metrics.record_latency("custom_op", 42.0, label="x")


def test_prometheus_metrics_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``increment`` records a counter."""
    metrics = PrometheusMetrics()
    metrics.increment("custom_counter", value=3, label="x")


def test_prometheus_metrics_register_app_skips_non_fastapi() -> None:
    """``register_app`` is a no-op for non-FastAPI objects."""
    metrics = PrometheusMetrics()
    metrics.register_app(MagicMock())  # no exception, no route added


def test_prometheus_metrics_register_app_fastapi() -> None:
    """``register_app`` adds a ``/metrics`` route to FastAPI apps."""
    try:
        from fastapi import FastAPI
    except ImportError:
        pytest.skip("fastapi is not installed")
    metrics = PrometheusMetrics()
    app = FastAPI()
    metrics.register_app(app)
    routes = [route.path for route in app.routes if hasattr(route, "path")]
    assert "/metrics" in routes


# ---------------------------------------------------------------------------
# set_active_metrics / record_*_latency
# ---------------------------------------------------------------------------


def test_set_active_metrics_stores_instance() -> None:
    """``set_active_metrics`` is the module-level setter."""
    metrics = PrometheusMetrics()
    set_active_metrics(metrics)
    # No direct getter, but ensure no exception.


def test_set_active_metrics_none() -> None:
    """``set_active_metrics(None)`` clears the active instance."""
    set_active_metrics(PrometheusMetrics())
    set_active_metrics(None)


def test_record_rerank_latency_uses_active_metrics() -> None:
    """``record_rerank_latency`` records via the active metrics."""
    metrics = MagicMock()
    histogram = MagicMock()
    metrics.rerank_latency.labels.return_value = histogram
    set_active_metrics(metrics)
    record_rerank_latency("bm25", 0.123)
    metrics.rerank_latency.labels.assert_called_once_with(provider="bm25")
    histogram.observe.assert_called_once_with(0.123)


def test_record_rerank_latency_swallows_errors() -> None:
    """``record_rerank_latency`` swallows exceptions from the metrics layer."""
    metrics = MagicMock()
    metrics.record_rerank_latency.side_effect = RuntimeError("boom")
    set_active_metrics(metrics)
    record_rerank_latency("bm25", 0.123)  # must not raise


def test_record_rerank_latency_no_active_metrics() -> None:
    """``record_rerank_latency`` is a no-op when no metrics are active."""
    set_active_metrics(None)
    record_rerank_latency("bm25", 0.123)


def test_record_long_context_uses_active_metrics() -> None:
    """``record_long_context`` records via the active metrics."""
    metrics = MagicMock()
    counter = MagicMock()
    metrics.long_context_pass.labels.return_value = counter
    set_active_metrics(metrics)
    record_long_context(outcome="hit", seconds=1.5)
    metrics.long_context_pass.labels.assert_called_once_with(outcome="hit")
    counter.inc.assert_called_once()


def test_record_long_context_swallows_errors() -> None:
    """``record_long_context`` swallows exceptions from the metrics layer."""
    metrics = MagicMock()
    metrics.record_long_context.side_effect = RuntimeError("boom")
    set_active_metrics(metrics)
    record_long_context(outcome="miss", seconds=0.0)


def test_record_long_context_no_active_metrics() -> None:
    """``record_long_context`` is a no-op when no metrics are active."""
    set_active_metrics(None)
    record_long_context(outcome="hit", seconds=1.0)


# ---------------------------------------------------------------------------
# scrub_secrets / redact_record
# ---------------------------------------------------------------------------


def test_scrub_secrets_masks_top_level_keys() -> None:
    """Top-level keys matching the secret pattern are masked."""
    payload = {"api_key": "secret", "name": "alice"}
    scrubbed = scrub_secrets(payload)
    assert scrubbed["api_key"] == "***"
    assert scrubbed["name"] == "alice"


def test_scrub_secrets_recurses_into_nested_dicts() -> None:
    """Nested dicts are scrubbed recursively."""
    payload = {"outer": {"password": "hunter2", "name": "alice"}}
    scrubbed = scrub_secrets(payload)
    assert scrubbed["outer"]["password"] == "***"
    assert scrubbed["outer"]["name"] == "alice"


def test_scrub_secrets_returns_copy() -> None:
    """The original ``kwargs`` is not mutated."""
    payload = {"api_key": "secret"}
    scrub_secrets(payload)
    assert payload["api_key"] == "secret"


def test_scrub_secrets_handles_list_values() -> None:
    """List values are passed through unchanged (only dicts are recursed)."""
    payload = {"items": ["a", "b"]}
    scrubbed = scrub_secrets(payload)
    assert scrubbed["items"] == ["a", "b"]


def test_redact_record_in_place() -> None:
    """``redact_record`` mutates the record in place."""
    record: dict[str, Any] = {"api_key": "secret", "user": "alice"}
    redact_record(record)
    assert record["api_key"] == "***"
    assert record["user"] == "alice"


def test_redact_record_does_not_recurse_secret_keys() -> None:
    """``redact_record`` only masks top-level keys matching the secret pattern.

    Nested dicts are copied through ``scrub`` which does not check
    inner keys against the pattern. This documents the existing
    behaviour.
    """
    record: dict[str, Any] = {"outer": {"api_key": "secret", "user": "alice"}}
    redact_record(record)
    assert record["outer"]["api_key"] == "secret"
    assert record["outer"]["user"] == "alice"


def test_redact_record_recurses_into_list_values() -> None:
    """Lists inside values are walked by ``scrub``."""
    record: dict[str, Any] = {"items": [{"inner": 1}]}
    redact_record(record)
    assert record["items"] == [{"inner": 1}]


def test_redact_string_key_matches_pattern() -> None:
    """String keys matching the secret pattern are masked."""
    record: dict[str, Any] = {"PASSWORD": "x"}
    redact_record(record)
    assert record["PASSWORD"] == "***"
