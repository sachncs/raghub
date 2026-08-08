"""Health service: aggregate liveness signals from key collaborators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raghub.services.diagnostics import (
    aggregate_status,
    emit_log,
    emit_metric,
    probe_embedder,
    probe_vector_store,
)
from raghub.types import JSONValue

if TYPE_CHECKING:
    from raghub.services.container import RagContainer


class Health:
    """Aggregate liveness signals from key collaborators."""

    def __init__(self, container: RagContainer) -> None:
        """Store the container reference."""
        self.container = container

    def log(self, message: str, **payload: JSONValue) -> None:
        """Emit a structured log event."""
        emit_log(self.container, message, **payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Record a latency metric."""
        emit_metric(self.container, name, started_at)

    def health(self) -> dict[str, object]:
        """Return a structured health report.

        The default implementation probes the vector store and the
        embedder, plus a static ``ok`` for the registry. The aggregate
        ``status`` is one of ``ok``, ``degraded``, ``down``.
        """
        self.log("health_check")
        components: dict[str, dict[str, object]] = {}
        components["vectorstore"] = probe_vector_store(self.container.vector_store)
        embedder = getattr(self.container, "embeddings", None)
        if embedder is not None:
            components["embedder"] = probe_embedder(embedder)
        components["registry"] = {"status": "ok"}
        return {
            "status": aggregate_status(components),
            "components": components,
        }


__all__ = ["Health"]
