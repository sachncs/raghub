"""Telemetry coverage tests.

Exercises the Langfuse-based telemetry path, the no-op recorders,
the secret-scrubbing helpers, and :class:`NoOpTelemetry`. The
``MetricsRegistry`` / ``PrometheusMetrics`` tests are gone in v0.7.0
because Prometheus has been removed from the codebase.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raghub.telemetry import (
    LangfuseTelemetryProvider,
    LoguruLogger,
    NoopSpan,
    NoOpTelemetry,
    RedactingTelemetry,
    Tracer,
    record_long_context,
    record_rerank_latency,
    redact_record,
    scrub_secrets,
    try_import_submodule,
)

# ---------------------------------------------------------------------------
# Hot-path recorders
# ---------------------------------------------------------------------------


def test_record_rerank_latency_silent_when_langfuse_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_rerank_latency`` is a silent no-op when Langfuse is unconfigured."""

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    record_rerank_latency("cohere", 0.1)  # does not raise


def test_record_long_context_silent_when_langfuse_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_long_context`` is a silent no-op when Langfuse is unconfigured."""

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    record_long_context(outcome="ran", seconds=0.05)


def test_record_rerank_latency_invokes_langfuse_score_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_rerank_latency`` calls ``client.score`` when Langfuse is configured."""

    fake_client = MagicMock()
    fake_client.score = MagicMock()
    monkeypatch.setattr("raghub.telemetry.tracing.langfuse_get_client", lambda: fake_client)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    record_rerank_latency("cohere", 0.1)
    fake_client.score.assert_called_once()
    kwargs = fake_client.score.call_args.kwargs
    assert kwargs["name"] == "raghub.rerank.latency"
    assert kwargs["value"] == 0.1
    assert kwargs["metadata"] == {"provider": "cohere"}


def test_record_long_context_invokes_langfuse_score_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_long_context`` calls ``client.score`` when Langfuse is configured."""

    fake_client = MagicMock()
    fake_client.score = MagicMock()
    monkeypatch.setattr("raghub.telemetry.tracing.langfuse_get_client", lambda: fake_client)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    record_long_context(outcome="ran", seconds=0.05)
    fake_client.score.assert_called_once()
    kwargs = fake_client.score.call_args.kwargs
    assert kwargs["name"] == "raghub.long_context.duration"
    assert kwargs["value"] == 0.05
    assert kwargs["metadata"] == {"outcome": "ran"}


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
    """redact_record handles lists by recursing into them."""

    record = {"items": [{"api_key": "x", "val": 1}]}
    redact_record(record)
    assert record["items"][0]["api_key"] == "***"
    assert record["items"][0]["val"] == 1


def test_redact_record_recurses_into_nested_dicts() -> None:
    """redact_record masks secret keys at any depth."""

    record = {"outer": {"inner": {"password": "x", "ok": "y"}}}
    redact_record(record)
    assert record["outer"]["inner"]["password"] == "***"
    assert record["outer"]["inner"]["ok"] == "y"


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
# LoguruLoggerAdapter basic path
# ---------------------------------------------------------------------------


def test_loguru_logger_warning_error_calls() -> None:
    """LoguruLoggerAdapter exposes warning + error methods."""

    logger = LoguruLoggerAdapter()
    logger.warning("warn message")
    logger.error("error message")


# ---------------------------------------------------------------------------
# RedactingTelemetry
# ---------------------------------------------------------------------------


def test_redacting_telemetry_wraps_inner() -> None:
    """RedactingTelemetry stores the inner provider for delegation."""

    inner = NoOpTelemetry()
    rt = RedactingTelemetry(inner)
    assert rt.inner is inner


def test_redacting_telemetry_span_passes_through() -> None:
    """RedactingTelemetry.start_span forwards to the inner provider.

    Uses :class:`NoOpTelemetry` (a real provider) as the inner so the
    test exercises the real delegation path. The wrapped span name
    surfaces through the NoOp span, so a regression that returned a
    different name or no span at all would be caught.
    """

    inner = NoOpTelemetry()
    rt = RedactingTelemetry(inner)
    span = rt.start_span("op")
    assert span.name == "op"
    rt.end_span(span)


def test_redacting_telemetry_info_scrubs_secret_keys() -> None:
    """``info`` is forwarded to the inner with secret-shaped keys replaced by ``***``.

    The inner provider captures the call's kwargs; we assert the
    secrets were redacted before forwarding. A regression that
    forwarded the raw value would fail this test.
    """

    captured: list[tuple[str, dict[str, object]]] = []

    class _CapturingTelemetry:
        def info(self, name: str, **kwargs: object) -> None:
            captured.append((name, dict(kwargs)))

    inner = _CapturingTelemetry()
    rt = RedactingTelemetry(inner)
    rt.info(
        "user login",
        username="alice",
        password="hunter2",
        api_key="sk-12345",
        nested={"authorization": "Bearer abc", "ok": "keep"},
    )

    assert captured, "Inner provider never received the call"
    name, kwargs = captured[0]
    assert name == "user login"
    assert kwargs["username"] == "alice"
    assert kwargs["password"] == "***"
    assert kwargs["api_key"] == "***"
    assert kwargs["nested"]["authorization"] == "***"
    assert kwargs["nested"]["ok"] == "keep"


def test_redacting_telemetry_record_latency_scrubs_labels() -> None:
    """``record_latency`` scrubs labels that match the secret-key regex."""

    captured: list[tuple[str, float, dict[str, object]]] = []

    class _CapturingTelemetry:
        def record_latency(self, name: str, value_ms: float, **labels: object) -> None:
            captured.append((name, value_ms, dict(labels)))

    rt = RedactingTelemetry(_CapturingTelemetry())
    rt.record_latency("op", 1.5, secret="raw", route="/v1")

    assert captured, "Inner provider never received the call"
    name, value_ms, labels = captured[0]
    assert name == "op"
    assert value_ms == 1.5
    assert labels["secret"] == "***"
    assert labels["route"] == "/v1"


def test_redacting_telemetry_record_tokens_passes_through() -> None:
    """``record_tokens`` forwards to the inner provider (no secret-shaped kwargs)."""

    captured: list[tuple[str, int, int, str]] = []

    class _CapturingTelemetry:
        def record_tokens(
            self,
            name: str,
            prompt_tokens: int,
            completion_tokens: int,
            model: str = "",
        ) -> None:
            captured.append((name, prompt_tokens, completion_tokens, model))

    rt = RedactingTelemetry(_CapturingTelemetry())
    rt.record_tokens("x", 1, 2, model="m")
    assert captured == [("x", 1, 2, "m")]


# ---------------------------------------------------------------------------
# Tracer (lightweight)
# ---------------------------------------------------------------------------


def test_tracer_construct() -> None:
    """Tracer can be built with default args."""

    tracer = Tracer()
    assert isinstance(tracer, Tracer)


# ---------------------------------------------------------------------------
# LangfuseTelemetryProvider configuration
# ---------------------------------------------------------------------------


def test_langfuse_provider_silent_when_unconfigured() -> None:
    """A Langfuse provider without credentials silently no-ops every call."""

    provider = LangfuseTelemetryProvider(public_key=None, secret_key=None)
    assert provider.client is None
    provider.info("anything")
    provider.warning("anything")
    provider.error("anything")
    provider.record_latency("x", 10.0)
    provider.increment("x", 1)


def test_langfuse_is_configured_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_configured`` reports env-var presence correctly."""

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert LangfuseTelemetryProvider.is_configured() is False
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert LangfuseTelemetryProvider.is_configured() is True
