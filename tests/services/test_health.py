"""Tests for ``raghub.services.health`` (Health service)."""

from __future__ import annotations

from types import SimpleNamespace

from raghub.services.health import Health


class HealthyVectorStore:
    def health(self) -> dict[str, object]:
        return {"status": "ok", "vector_count": 10}


class DegradedVectorStore:
    def health(self) -> dict[str, object]:
        return {"status": "stale"}


class StubEmbedder:
    model_name = "test-embedder"

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_health_reports_ok_when_every_component_healthy() -> None:
    """``Health.health`` returns status='ok' when every probe reports healthy."""

    container = SimpleNamespace(vector_store=HealthyVectorStore(), embeddings=StubEmbedder())
    h = Health(container)
    report = h.health()
    assert report.status == "ok"
    assert "components" in {c for c in (report.components,)}
    assert "vectorstore" in report.components
    assert "embedder" in report.components
    assert "registry" in report.components
    assert report.components["vectorstore"].status == "ok"
    assert report.components["embedder"].status == "ok"
    assert report.components["registry"].status == "ok"


def test_health_reports_degraded_when_vector_store_stale() -> None:
    """``Health.health`` returns 'degraded' when a vector store probe fails."""

    container = SimpleNamespace(vector_store=DegradedVectorStore(), embeddings=StubEmbedder())
    h = Health(container)
    report = h.health()
    assert report.status == "degraded"
    assert report.components["vectorstore"].status == "degraded"


def test_health_skips_embedder_probe_when_missing() -> None:
    """``Health.health`` does not probe a missing embedder."""

    container = SimpleNamespace(vector_store=HealthyVectorStore())
    h = Health(container)
    report = h.health()
    assert "embedder" not in report.components
    assert report.status == "ok"


def test_health_reports_vector_store_down_when_health_raises() -> None:
    """``Health.health`` propagates 'unknown' from a probe without health() method."""

    class NoHealthStore:
        pass

    container = SimpleNamespace(vector_store=NoHealthStore(), embeddings=StubEmbedder())
    h = Health(container)
    report = h.health()
    assert report.components["vectorstore"].status == "unknown"
    # 'unknown' maps to 'degraded' under aggregate_status.
    assert report.status == "degraded"


def test_health_emits_health_check_log_event() -> None:
    """``Health.health`` calls ``container.logger.info`` with 'health_check'."""

    captured: list[tuple[str, dict[str, object]]] = []

    class TestLogger:
        def info(self, message: str, **kwargs: object) -> None:
            captured.append((message, kwargs))

    container = SimpleNamespace(
        vector_store=HealthyVectorStore(),
        embeddings=StubEmbedder(),
        logger=TestLogger(),
    )
    Health(container).health()
    assert any(msg == "health_check" for msg, _ in captured)


def test_health_log_delegates_to_emit_log() -> None:
    """``Health.log`` forwards to ``emit_log`` with the given payload."""

    captured: list[tuple[str, dict[str, object]]] = []

    class TestLogger:
        def info(self, message: str, **kwargs: object) -> None:
            captured.append((message, kwargs))

    container = SimpleNamespace(logger=TestLogger())
    Health(container).log("custom.event", user_id="alice", action="ping")
    assert captured == [("custom.event", {"extra": {"user_id": "alice", "action": "ping"}})]


def test_health_emit_metric_delegates_to_emit_metric() -> None:
    """``Health.emit_metric`` forwards to ``emit_metric`` with elapsed time."""

    import time as _time

    captured: list[tuple[str, float]] = []

    class TestMetrics:
        def record_latency(self, name: str, value_ms: float) -> None:
            captured.append((name, value_ms))

    container = SimpleNamespace(metrics=TestMetrics())
    started = _time.perf_counter()
    Health(container).emit_metric("ingest.run", started)
    assert len(captured) == 1
    assert captured[0][0] == "ingest.run"
    assert captured[0][1] >= 0
