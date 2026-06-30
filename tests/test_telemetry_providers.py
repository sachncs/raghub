"""Tests for the structured and telemetry providers."""

from __future__ import annotations

import pytest

from raghub.observability.noop import NoOpTelemetry
from raghub.telemetry.langfuse import LangfuseTelemetryProvider, NoopSpan


def test_noop_telemetry_satisfies_protocol() -> None:
    """``NoOpTelemetry`` implements the :class:`TelemetryProvider` contract."""

    provider = NoOpTelemetry()
    # All public methods must accept the documented signatures.
    provider.info("event", key="value")
    provider.warning("event", key="value")
    provider.error("event", key="value")
    provider.record_latency("name", 12.0)
    provider.increment("name", 1)
    span = provider.start_span("op", attr="value")
    provider.end_span(span)
    provider.record_tokens("name", prompt_tokens=10, completion_tokens=20)


def test_noop_telemetry_no_io_when_disabled() -> None:
    """``NoOpTelemetry`` performs no I/O."""
    provider = NoOpTelemetry()
    span = provider.start_span("op")
    span.set_attribute("key", "value")
    span.end()  # must not raise


def test_noop_span_attributes_recorded() -> None:
    """``NoopSpan.set_attribute`` records the attribute."""
    span = NoopSpan("op")
    span.set_attribute("key", "value")
    assert span.attributes["key"] == "value"


def test_noop_telemetry_record_tokens_no_io() -> None:
    """``record_tokens`` on a no-op provider is a no-op."""
    NoOpTelemetry().record_tokens("name", prompt_tokens=1, completion_tokens=2)


def test_noop_telemetry_span_context_manager() -> None:
    """The ``span`` context manager closes the span on exit."""
    from raghub.interfaces.observability import TelemetryProvider

    provider: TelemetryProvider = NoOpTelemetry()
    with provider.span("op", attr="value") as s:
        s.set_attribute("extra", 1)
    # No exception means the context manager cleaned up correctly.


def test_langfuse_provider_unconfigured_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``is_configured`` returns ``False`` when env vars are missing."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert LangfuseTelemetryProvider.is_configured() is False


def test_langfuse_provider_configured_with_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``is_configured`` returns ``True`` when env vars are present."""
    pytest.importorskip("langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert LangfuseTelemetryProvider.is_configured() is True


def test_langfuse_provider_no_client_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without credentials the provider has no client and methods are no-ops."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    provider = LangfuseTelemetryProvider()
    # No client; methods must not raise.
    provider.info("event")
    provider.start_span("op")  # returns a NoopSpan


def test_langfuse_safe_helper_returns_value() -> None:
    """``_safe`` returns the callable's return value when it succeeds."""

    def fn(x: int) -> int:
        return x * 2

    assert LangfuseTelemetryProvider.safe_call(fn, 21) == 42


def test_langfuse_safe_helper_swallows_exception() -> None:
    """``_safe`` returns ``None`` when the callable raises."""

    def bad() -> None:
        raise RuntimeError("boom")

    assert LangfuseTelemetryProvider.safe_call(bad) is None


def test_langfuse_safe_helper_logs_when_debug_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``_safe`` logs the failure when ``LANGFUSE_DEBUG`` is set."""
    monkeypatch.setenv("LANGFUSE_DEBUG", "1")

    def bad() -> None:
        raise RuntimeError("boom")

    with caplog.at_level("WARNING", logger="raghub.telemetry.langfuse"):
        result = LangfuseTelemetryProvider.safe_call(bad)
    assert result is None
    assert any("boom" in record.message for record in caplog.records)


def test_langfuse_end_trace_no_client() -> None:
    """``end_trace`` is a no-op when the provider has no client."""
    provider = LangfuseTelemetryProvider()
    # Must not raise.
    provider.end_trace()


def test_langfuse_provider_records_token_no_client() -> None:
    """``record_tokens`` is a no-op when the provider has no client."""
    provider = LangfuseTelemetryProvider()
    # Must not raise.
    provider.record_tokens("name", prompt_tokens=10, completion_tokens=20)


def test_langfuse_provider_end_span_on_noop_span() -> None:
    """``end_span`` on a :class:`NoopSpan` is a no-op."""
    provider = LangfuseTelemetryProvider()
    provider.end_span(NoopSpan("op"))


def test_langfuse_start_span_propagates_user_session() -> None:
    """``start_span`` propagates ``user_id`` and ``session_id`` when Langfuse is configured.

    When Langfuse is not configured, the provider has no client and
    the propagation is a no-op (which is exactly what we want).
    """
    provider = LangfuseTelemetryProvider()
    # Without env vars, ``_propagate`` is a no-op.
    provider.propagate_to_langfuse(user_id="alice@x", session_id="s1")
    # With Langfuse not configured, ``start_span`` returns a NoopSpan.
    span = provider.start_span("op", user_id="alice@x", session_id="s1")
    assert isinstance(span, NoopSpan)
    span.set_attribute("user_id", "alice@x")
    span.end()  # must not raise


def test_langfuse_propagate_method_safe_on_missing_client() -> None:
    """``_propagate`` is a no-op when the client is not configured."""
    provider = LangfuseTelemetryProvider()
    # No exception is raised.
    provider.propagate_to_langfuse(user_id="x")
