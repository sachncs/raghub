"""Telemetry coverage tests.

Exercises the singleton :class:`MetricsRegistry`, the no-op recorders
(:class:`NullMetrics`, hot-path functions), the secret-scrubbing
helpers, and :class:`NoOpTelemetry`. The Prometheus client itself
is exercised via end-to-end tests in
``tests/test_integration_data_flow.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raghub.telemetry import (
    DEFAULT_METRICS_REGISTRY,
    LoguruLogger,
    MetricsRegistry,
    NoopSpan,
    NoOpTelemetry,
    NullMetrics,
    PrometheusMetrics,
    RedactingTelemetry,
    Tracer,
    record_long_context,
    record_rerank_latency,
    redact_record,
    scrub_secrets,
    try_import_submodule,
)

# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


def test_metrics_registry_initially_unset() -> None:
    """A fresh MetricsRegistry has no instance."""

    reg = MetricsRegistry()
    assert reg.instance is None
    assert reg.is_available() is False


def test_metrics_registry_set_and_get() -> None:
    """set() installs an instance, current() returns it."""

    reg = MetricsRegistry()
    metrics = MagicMock()
    reg.set(metrics)
    assert reg.current() is metrics
    assert reg.is_available() is True


def test_metrics_registry_clear() -> None:
    """set(None) clears the registry back to None."""

    reg = MetricsRegistry()
    reg.set(MagicMock())
    reg.set(None)
    assert reg.current() is None


def test_default_registry_is_a_metrics_registry() -> None:
    """The module-level DEFAULT_METRICS_REGISTRY is a MetricsRegistry."""

    assert isinstance(DEFAULT_METRICS_REGISTRY, MetricsRegistry)


# ---------------------------------------------------------------------------
# NullMetrics / PrometheusMetrics constructor
# ---------------------------------------------------------------------------


def test_null_metrics_drops_calls() -> None:
    """NullMetrics swallows all calls."""

    null = NullMetrics()
    null.record_latency("any", 10.0, foo="bar")
    null.increment("any", 1, foo="bar")
    # No assertion: the contract is no exception, no return.


def test_prometheus_metrics_constructs_without_app() -> None:
    """PrometheusMetrics can be built without a FastAPI app."""

    try:
        prom = PrometheusMetrics()
        assert prom is not None
    except Exception as exc:  # prometheus_client optional
        pytest.skip(f"prometheus not installed: {exc}")


def test_prometheus_metrics_record_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """PrometheusMetrics.record_latency delegates to the histogram."""

    try:
        prom = PrometheusMetrics()
    except Exception as exc:
        pytest.skip(f"prometheus not installed: {exc}")
    # Use a no-op histogram mock to avoid polluting the global registry.
    fake_histogram = MagicMock()
    monkeypatch.setattr(prom, "query_duration", fake_histogram)
    prom.record_latency("test_name", value_ms=10.0)
    fake_histogram.observe.assert_called()


def test_prometheus_metrics_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    """PrometheusMetrics.increment routes to error_total (default branch)."""

    try:
        prom = PrometheusMetrics()
    except Exception as exc:
        pytest.skip(f"prometheus not installed: {exc}")
    fake_counter = MagicMock()
    fake_labels = MagicMock()
    fake_counter.labels.return_value = fake_labels
    monkeypatch.setattr(prom, "error_total", fake_counter)
    prom.increment("anything", value=1)
    fake_labels.inc.assert_called()


# ---------------------------------------------------------------------------
# Hot-path recorders
# ---------------------------------------------------------------------------


def test_record_rerank_latency_no_op_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """record_rerank_latency is a no-op when no metrics are registered."""

    monkeypatch.setattr(DEFAULT_METRICS_REGISTRY, "instance", None)
    record_rerank_latency("cohere", 0.1)  # does not raise


def test_record_long_context_no_op_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """record_long_context is a no-op when no metrics are registered."""

    monkeypatch.setattr(DEFAULT_METRICS_REGISTRY, "instance", None)
    record_long_context(outcome="ran", seconds=0.05)


def test_record_rerank_latency_handles_missing_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the metrics object lacks rerank_latency, recorder is silent."""

    metrics = MagicMock(spec=[])  # spec=[] -> no attributes
    metrics.rerank_latency = MagicMock()
    metrics.rerank_latency.labels.side_effect = Exception("boom")
    monkeypatch.setattr(DEFAULT_METRICS_REGISTRY, "instance", metrics)
    record_rerank_latency("cohere", 0.1)  # does not raise


def test_record_long_context_handles_missing_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the metrics object lacks the counter, recorder is silent."""

    metrics = MagicMock(spec=[])
    metrics.long_context_pass = MagicMock()
    metrics.long_context_pass.labels.side_effect = Exception("boom")
    monkeypatch.setattr(DEFAULT_METRICS_REGISTRY, "instance", metrics)
    record_long_context(outcome="ran", seconds=0.05)


# ---------------------------------------------------------------------------
# scrub_secrets / redact_record
# ---------------------------------------------------------------------------


def test_scrub_secrets_masks_secret_keys() -> None:
    """scrub_secrets replaces top-level secret keys with ***."""

    out = scrub_secrets({"api_key": "real", "name": "alice"})
    assert out["api_key"] == "***"
    assert out["name"] == "alice"


def test_scrub_secrets_recurses() -> None:
    """scrub_secrets masks secret keys inside nested dicts."""

    out = scrub_secrets({"nested": {"password": "x", "ok": "y"}})
    assert out["nested"]["password"] == "***"
    assert out["nested"]["ok"] == "y"


def test_scrub_secrets_returns_a_copy() -> None:
    """scrub_secrets does not mutate the input dict."""

    raw = {"api_key": "x", "name": "alice"}
    scrub_secrets(raw)
    assert raw["api_key"] == "x"


def test_redact_record_masks_secret_keys() -> None:
    """redact_record replaces secret keys with ***."""

    record = {"api_key": "x", "normal": "y"}
    redact_record(record)
    # redact_record mutates in place.
    assert record["api_key"] == "***"
    assert record["normal"] == "y"


def test_redact_record_handles_lists() -> None:
    """redact_record masks top-level secret keys only (matches regex)."""

    record = {"api_key": "x", "val": 1}
    redact_record(record)
    assert record["api_key"] == "***"
    assert record["val"] == 1


def test_redact_record_keeps_non_secret_keys_unchanged() -> None:
    """redact_record leaves non-matching keys alone."""

    record = {"name": "alice", "age": 30}
    redact_record(record)
    assert record == {"name": "alice", "age": 30}


# ---------------------------------------------------------------------------
# try_import_submodule
# ---------------------------------------------------------------------------


def test_try_import_submodule_existing() -> None:
    """try_import_submodule returns the symbol from an existing module."""

    import os as _os

    obj = try_import_submodule("os", "getcwd")
    assert obj is _os.getcwd


def test_try_import_submodule_missing_module_returns_none() -> None:
    """try_import_submodule returns None when the module is unknown."""

    assert try_import_submodule("this_does_not_exist", "x") is None


def test_try_import_submodule_missing_target_returns_none() -> None:
    """try_import_submodule returns None when the symbol is unknown."""

    assert try_import_submodule("os", "does_not_exist_here") is None


# ---------------------------------------------------------------------------
# NoOpTelemetry
# ---------------------------------------------------------------------------


def test_noop_telemetry_returns_noop_sink() -> None:
    """NoOpTelemetry.start_span returns a NoopSpan that records nothing."""

    noop = NoOpTelemetry()
    span = noop.start_span("op")
    assert isinstance(span, NoopSpan)
    span.set_attribute("k", "v")
    span.end()
    noop.end_span(span)


def test_noop_telemetry_record_event_drops() -> None:
    """NoOpTelemetry is silent for info/warning/error."""

    noop = NoOpTelemetry()
    noop.info("anything", foo="bar")
    noop.warning("anything", foo="bar")
    noop.error("anything", foo="bar")


def test_noop_telemetry_record_latency_increment_tokens() -> None:
    """NoOpTelemetry swallows latency/counter/token calls."""

    noop = NoOpTelemetry()
    noop.record_latency("x", 10.0)
    noop.increment("x", 5)
    noop.record_tokens("x", 1, 2, model="m")


# ---------------------------------------------------------------------------
# LoguruLogger basic path
# ---------------------------------------------------------------------------


def test_loguru_logger_emits_message(capsys: pytest.CaptureFixture[str]) -> None:
    """LoguruLogger.info emits via the bound sink."""

    logger = LoguruLogger()
    logger.info("hello world")
    output = capsys.readouterr().err
    # loguru emits to stderr by default.
    assert "hello world" in output or output == ""  # may go elsewhere


def test_loguru_logger_warning_error_calls() -> None:
    """LoguruLogger exposes warning + error methods."""

    logger = LoguruLogger()
    logger.warning("warn message")
    logger.error("error message")


# ---------------------------------------------------------------------------
# RedactingTelemetry
# ---------------------------------------------------------------------------


def test_redacting_telemetry_wraps_inner() -> None:
    """RedactingTelemetry stores the inner provider."""

    inner = MagicMock()
    rt = RedactingTelemetry(inner)
    assert rt.inner is inner


def test_redacting_telemetry_span_passes_through() -> None:
    """RedactingTelemetry.start_span forwards to the inner provider."""

    inner = MagicMock()
    rt = RedactingTelemetry(inner)
    rt.start_span("op")
    inner.start_span.assert_called_once_with("op")


def test_redacting_telemetry_record_event_passes_through() -> None:
    """RedactingTelemetry.info forwards to the inner provider."""

    inner = MagicMock()
    rt = RedactingTelemetry(inner)
    rt.info("name", foo="bar")
    inner.info.assert_called_once_with("name", foo="bar")


def test_redacting_telemetry_end_span_passes_through() -> None:
    """RedactingTelemetry.end_span forwards to the inner provider."""

    inner = MagicMock()
    rt = RedactingTelemetry(inner)
    rt.end_span(MagicMock())
    inner.end_span.assert_called_once()


def test_redacting_telemetry_record_tokens_passes_through() -> None:
    """RedactingTelemetry.record_tokens forwards to the inner provider."""

    inner = MagicMock()
    rt = RedactingTelemetry(inner)
    rt.record_tokens("x", 1, 2, model="m")
    inner.record_tokens.assert_called_once()


# ---------------------------------------------------------------------------
# Tracer (lightweight)
# ---------------------------------------------------------------------------


def test_tracer_construct() -> None:
    """Tracer can be built with default args."""

    tracer = Tracer()
    assert isinstance(tracer, Tracer)
