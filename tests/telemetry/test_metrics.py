"""Tests for ``raghub.telemetry.metrics`` (NoOpTelemetry / NoopSpan)."""

from __future__ import annotations

from raghub.telemetry.metrics import NoopSpan, NoOpTelemetry


def test_noop_span_records_name() -> None:
    """``NoopSpan`` stores the supplied name."""
    span = NoopSpan(name="ingest.run")
    assert span.name == "ingest.run"


def test_noop_span_end_is_silent() -> None:
    """``NoopSpan.end`` performs no work and returns None."""
    span = NoopSpan(name="x")
    assert span.end() is None


def test_noop_span_set_attribute_records_value() -> None:
    """``NoopSpan.set_attribute`` records the value for later inspection."""

    span = NoopSpan(name="x")
    span.set_attribute("status", "ok")
    span.set_attribute("count", 42)
    assert span.attributes == {"status": "ok", "count": 42}


def test_noop_span_attributes_returns_independent_copy() -> None:
    """Mutating the returned dict does not affect the span's internal state."""

    span = NoopSpan(name="x")
    span.set_attribute("k", "v")
    snapshot = span.attributes
    snapshot["k"] = "mutated"
    assert span.attrs == {"k": "v"}


def test_noop_telemetry_info_warning_error_are_silent() -> None:
    """Log methods on NoOpTelemetry all return None and do not raise."""

    provider = NoOpTelemetry()
    assert provider.info("any", code=42) is None
    assert provider.warning("any") is None
    assert provider.error("any") is None


def test_noop_telemetry_record_latency_is_silent() -> None:
    """``record_latency`` accepts arbitrary kwargs without raising."""

    provider = NoOpTelemetry()
    assert provider.record_latency("ingest", 12.5, route="/v1") is None


def test_noop_telemetry_increment_is_silent() -> None:
    """``increment`` accepts arbitrary kwargs without raising."""

    provider = NoOpTelemetry()
    assert provider.increment("counter", 5, route="/v1") is None


def test_noop_telemetry_start_span_returns_noop_span() -> None:
    """``start_span`` returns a :class:`NoopSpan` with the supplied name."""

    provider = NoOpTelemetry()
    span = provider.start_span("ingest")
    assert isinstance(span, NoopSpan)
    assert span.name == "ingest"


def test_noop_telemetry_end_span_is_silent() -> None:
    """``end_span`` accepts any span and returns None."""

    provider = NoOpTelemetry()
    assert provider.end_span(NoopSpan("x")) is None


def test_noop_telemetry_record_tokens_is_silent() -> None:
    """``record_tokens`` accepts arbitrary inputs without raising."""

    provider = NoOpTelemetry()
    assert provider.record_tokens("ingest", 10, 20, model="m") is None


def test_noop_telemetry_span_context_manager() -> None:
    """``NoOpTelemetry.span`` returns a context manager that yields then closes."""

    provider = NoOpTelemetry()
    with provider.span("ingest", route="/v1") as span:
        assert isinstance(span, NoopSpan)
        assert span.name == "ingest"
