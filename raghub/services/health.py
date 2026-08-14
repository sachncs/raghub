"""Health service: aggregate liveness signals from key collaborators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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


@dataclass(slots=True, frozen=True)
class ComponentHealth:
    """Liveness signal from a single collaborator.

    Attributes:
        status: One of ``"ok"``, ``"degraded"``, ``"down"``.
        extra: Component-specific diagnostic data (latency, row
            counts, error messages, etc.).

    """

    status: str = "ok"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class HealthReport:
    """The aggregate health snapshot returned by :meth:`Health.health`.

    Attributes:
        status: One of ``"ok"``, ``"degraded"``, ``"down"`` (aggregate).
        components: Per-collaborator :class:`ComponentHealth` keyed by
            the collaborator name (``"vectorstore"``, ``"embedder"``,
            ``"registry"``, etc.).

    """

    status: str = "ok"
    components: dict[str, ComponentHealth] = field(default_factory=dict)


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

    def health(self) -> HealthReport:
        """Return a structured :class:`HealthReport`.

        The default implementation probes the vector store and the
        embedder, plus a static ``ok`` for the registry. The aggregate
        ``status`` is one of ``ok``, ``degraded``, ``down``.
        """
        self.log("health_check")
        components: dict[str, ComponentHealth] = {}
        components["vectorstore"] = ComponentHealth(**probe_vector_store(self.container.vector_store))
        embedder = getattr(self.container, "embeddings", None)
        if embedder is not None:
            components["embedder"] = ComponentHealth(**probe_embedder(embedder))
        components["registry"] = ComponentHealth(status="ok")
        probe_dicts = {
            name: {"status": comp.status, **comp.extra}
            for name, comp in components.items()
        }
        return HealthReport(
            status=aggregate_status(probe_dicts),
            components=components,
        )


__all__ = ["ComponentHealth", "Health", "HealthReport"]
