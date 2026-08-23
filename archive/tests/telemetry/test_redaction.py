"""Tests for ``raghub.telemetry.redaction``."""

from __future__ import annotations

from typing import Any

from raghub.telemetry.metrics import NoOpTelemetry
from raghub.telemetry.redaction import (
    SECRET_KEY_RE,
    RedactingTelemetry,
    redact_record,
    scrub_secrets,
)


def test_secret_key_re_matches_password() -> None:
    """``SECRET_KEY_RE`` matches any key containing 'password'."""

    assert SECRET_KEY_RE.search("password") is not None
    assert SECRET_KEY_RE.search("user_password") is not None
    assert SECRET_KEY_RE.search("PASSWORD") is not None


def test_secret_key_re_matches_other_secret_shapes() -> None:
    """``SECRET_KEY_RE`` covers api_key, token, jwt, authorization."""

    for key in ("api_key", "apikey", "access_token", "refresh_token", "jwt", "authorization"):
        assert SECRET_KEY_RE.search(key) is not None, key


def test_secret_key_re_ignores_non_secret_keys() -> None:
    """``SECRET_KEY_RE`` does not match non-secret keys."""

    for key in ("user_id", "email", "route", "company", "duration_ms"):
        assert SECRET_KEY_RE.search(key) is None, key


def test_scrub_secrets_masks_top_level_secrets() -> None:
    """``scrub_secrets`` masks values whose key matches the secret pattern."""

    scrubbed = scrub_secrets({"password": "secret123", "email": "alice@example.com"})
    assert scrubbed == {"password": "***", "email": "alice@example.com"}


def test_scrub_secrets_recurses_into_nested_dicts() -> None:
    """``scrub_secrets`` recurses into nested dicts to find secret keys."""

    scrubbed = scrub_secrets(
        {
            "user": {"api_key": "sk-abc", "name": "alice"},
            "metadata": {"company": "acme"},
        }
    )
    assert scrubbed == {
        "user": {"api_key": "***", "name": "alice"},
        "metadata": {"company": "acme"},
    }


def test_scrub_secrets_does_not_mutate_input() -> None:
    """``scrub_secrets`` returns a fresh dict; the input is unchanged."""

    original = {"password": "secret", "email": "alice@example.com"}
    scrubbed = scrub_secrets(original)
    assert original == {"password": "secret", "email": "alice@example.com"}
    assert scrubbed is not original


def test_scrub_secrets_preserves_non_string_values() -> None:
    """``scrub_secrets`` leaves ints, bools, floats unchanged."""

    scrubbed = scrub_secrets({"count": 42, "enabled": True, "ratio": 0.5})
    assert scrubbed == {"count": 42, "enabled": True, "ratio": 0.5}


def test_redact_record_redacts_in_place() -> None:
    """``redact_record`` mutates the supplied dict."""

    record: dict[str, Any] = {"api_key": "sk-abc", "email": "alice@example.com"}
    redact_record(record)
    assert record == {"api_key": "***", "email": "alice@example.com"}


def test_redact_record_recurses_into_lists() -> None:
    """``redact_record`` walks list elements and scrubs each."""

    record: dict[str, Any] = {
        "events": [{"api_key": "sk-1"}, {"api_key": "sk-2"}],
        "metadata": "plain",
    }
    redact_record(record)
    assert record == {
        "events": [{"api_key": "***"}, {"api_key": "***"}],
        "metadata": "plain",
    }


def test_redacting_telemetry_wraps_inner() -> None:
    """``RedactingTelemetry`` stores the inner provider for delegation."""

    inner = NoOpTelemetry()
    rt = RedactingTelemetry(inner)
    assert rt.inner is inner


def test_redacting_telemetry_scrubs_before_forwarding_info() -> None:
    """``RedactingTelemetry.info`` masks secret kwargs before forwarding."""

    captured: list[dict[str, Any]] = []

    class Capture:
        def info(self, message: str, **kwargs: Any) -> None:
            captured.append(kwargs)

    rt = RedactingTelemetry(Capture())
    rt.info("test.event", password="secret", email="alice@example.com")
    assert captured == [{"password": "***", "email": "alice@example.com"}]


def test_redacting_telemetry_scrubs_before_forwarding_warning() -> None:
    """``RedactingTelemetry.warning`` masks secret kwargs."""

    captured: list[dict[str, Any]] = []

    class Capture:
        def warning(self, message: str, **kwargs: Any) -> None:
            captured.append(kwargs)

    rt = RedactingTelemetry(Capture())
    rt.warning("auth.failed", authorization="Bearer xyz", user_id="alice")
    assert captured == [{"authorization": "***", "user_id": "alice"}]


def test_redacting_telemetry_scrubs_record_latency_labels() -> None:
    """``RedactingTelemetry.record_latency`` scrubs the labels dict."""

    captured: list[dict[str, Any]] = []

    class Capture:
        def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
            captured.append(labels)

    rt = RedactingTelemetry(Capture())
    rt.record_latency("ingest", 12.5, jwt="abc", route="/v1")
    assert captured == [{"jwt": "***", "route": "/v1"}]


def test_redacting_telemetry_scrubs_span_attributes() -> None:
    """``RedactingTelemetry.start_span`` scrubs the attrs dict."""

    received: dict[str, Any] = {}

    class StubSpan:
        def __init__(self, attrs: dict[str, Any]) -> None:
            received.update(attrs)

    class StubProvider:
        def start_span(self, name: str, **attrs: Any) -> Any:
            return StubSpan(attrs)

    rt = RedactingTelemetry(StubProvider())
    rt.start_span("x", secret="shh", user="alice")
    assert received == {"secret": "***", "user": "alice"}
