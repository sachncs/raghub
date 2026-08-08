"""Tests for ``raghub.services.diagnostics``."""

from __future__ import annotations

import pytest

from raghub.config import Settings
from raghub.services.diagnostics import (
    aggregate_status,
    build_models,
    emit_log,
    emit_metric,
    missing_doc,
    parse_users,
    probe_embedder,
    probe_vector_store,
    seed_blocked,
)


def test_emit_log_routes_to_container_logger() -> None:
    """``emit_log`` calls the container's logger.info method."""

    captured: list[tuple[str, dict[str, object]]] = []

    class TestLogger:
        def info(self, message: str, **kwargs: object) -> None:
            # emit_log forwards payload as the 'extra' kwarg
            payload = kwargs.pop("extra", {})
            captured.append((message, payload))

    container = type("C", (), {"logger": TestLogger()})()
    emit_log(container, "test.event", code=42)
    assert captured == [("test.event", {"code": 42})]


def test_emit_log_skips_when_container_lacks_logger() -> None:
    """``emit_log`` is a no-op when the container has no logger."""
    container = type("C", (), {})()
    emit_log(container, "test.event")  # must not raise


def test_emit_metric_calls_record_latency() -> None:
    """``emit_metric`` invokes ``metrics.record_latency`` with elapsed ms."""

    captured: list[tuple[str, float]] = []

    class TestMetrics:
        def record_latency(self, name: str, value_ms: float) -> None:
            captured.append((name, value_ms))

    container = type("C", (), {"metrics": TestMetrics()})()
    import time as _time

    started = _time.perf_counter()
    _time.sleep(0.001)
    emit_metric(container, "ingest", started)
    assert len(captured) == 1
    assert captured[0][0] == "ingest"
    assert captured[0][1] > 0


def test_missing_doc_raises_ingestion_error() -> None:
    """``missing_doc`` raises :class:`IngestionError` for an unknown id."""
    from raghub.errors import IngestionError

    with pytest.raises(IngestionError, match="Unknown document id: missing"):
        missing_doc("missing")


def test_probe_vector_store_returns_unknown_when_no_health() -> None:
    """``probe_vector_store`` reports 'unknown' when collaborator lacks health."""

    assert probe_vector_store(object()) == {
        "status": "unknown",
        "detail": "no health() method",
    }


def test_probe_vector_store_passes_through_health_dict() -> None:
    """``probe_vector_store`` echoes a healthy payload."""

    class StubStore:
        def health(self) -> dict[str, object]:
            return {"status": "ok", "vector_count": 100}

    assert probe_vector_store(StubStore()) == {
        "status": "ok",
        "vector_count": 100,
    }


def test_probe_vector_store_flags_degraded_status() -> None:
    """``probe_vector_store`` downgrades unknown statuses to 'degraded'."""

    class StubStore:
        def health(self) -> dict[str, object]:
            return {"status": "stale"}

    assert probe_vector_store(StubStore())["status"] == "degraded"


def test_probe_embedder_reports_unknown_for_none() -> None:
    """``probe_embedder`` reports 'unknown' when no embedder is configured."""
    assert probe_embedder(None)["status"] == "unknown"


def test_probe_embedder_reports_ok_when_embed_text_returns_list() -> None:
    """``probe_embedder`` reports 'ok' with the model name and dimension."""

    class StubEmbedder:
        model_name = "test-model"

        def embed_text(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    payload = probe_embedder(StubEmbedder())
    assert payload["status"] == "ok"
    assert payload["dimension"] == 3
    assert payload["model"] == "test-model"


def test_aggregate_status_prefers_down_over_degraded() -> None:
    """``aggregate_status`` short-circuits to 'down' when any probe fails."""

    assert (
        aggregate_status(
            {
                "store": {"status": "down"},
                "embedder": {"status": "ok"},
            }
        )
        == "down"
    )


def test_aggregate_status_prefers_degraded_over_ok() -> None:
    """``aggregate_status`` reports 'degraded' when nothing is 'down'."""

    assert (
        aggregate_status(
            {
                "store": {"status": "degraded"},
                "embedder": {"status": "ok"},
            }
        )
        == "degraded"
    )


def test_aggregate_status_returns_ok_when_all_ok() -> None:
    """``aggregate_status`` returns 'ok' when every probe reports ok."""

    assert (
        aggregate_status(
            {
                "store": {"status": "ok"},
                "embedder": {"status": "healthy"},
            }
        )
        == "ok"
    )


def test_seed_blocked_skips_production() -> None:
    """``seed_blocked`` returns True when environment is 'production'."""

    settings = Settings(environment="production")
    assert seed_blocked(settings) is True


def test_seed_blocked_skips_wildcard_cors(monkeypatch: pytest.MonkeyPatch) -> None:
    """``seed_blocked`` returns True when CORS_ORIGINS is a wildcard."""

    monkeypatch.setenv("CORS_ORIGINS", "*")
    settings = Settings(environment="development")
    assert seed_blocked(settings) is True


def test_seed_blocked_allows_when_development_and_safe_cors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``seed_blocked`` returns False for development + non-wildcard CORS."""

    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    settings = Settings(environment="development")
    assert seed_blocked(settings) is False


def test_parse_users_returns_dict_for_json() -> None:
    """``parse_users`` parses a JSON env-var payload."""

    payload = '{"alice": {"password": "x", "companies": ["a"]}}'
    parsed = parse_users(payload)
    assert isinstance(parsed, dict)
    assert "alice" in parsed