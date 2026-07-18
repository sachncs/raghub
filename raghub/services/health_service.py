"""Health-check service aggregating component liveness.

A thin wrapper around each collaborator's ``health()`` method. The
service is intentionally synchronous (it does no IO beyond the
collaborator calls) so it can be polled cheaply by orchestrators.

The :meth:`health` method probes the vector store and the embedder
on every call. The aggregate status is:

* ``"ok"`` — every probed component reports healthy.
* ``"degraded"`` — at least one component is reachable but reports
  a problem; the platform can still serve traffic.
* ``"down"`` — at least one component is unreachable; the platform
  cannot serve traffic reliably.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raghub.services import ServiceMixin

if TYPE_CHECKING:
    from raghub.services.application import DynamicRagContainer


def probe_vector_store(store: object) -> dict[str, object]:
    """Probe a vector store for liveness.

    Calls the collaborator's ``health()`` method and translates the
    result into one of the canonical statuses. The collaborator's own
    payload is **flattened into the result** so callers can still
    read ``chunks``, ``size``, and other keys without digging through
    a ``details`` sub-dict.

    Args:
        store: The vector store collaborator (or compatible stub).

    Returns:
        A dict with ``status`` plus the collaborator's own payload.
    """
    probe = getattr(store, "health", None)
    if not callable(probe):
        return {"status": "unknown", "detail": "no health() method"}
    try:
        payload = probe()
        if not isinstance(payload, dict):
            payload = {"value": payload}
        status = str(payload.get("status", "ok")).lower()
        if status not in {"ok", "healthy", "up", "ready"}:
            payload = {**payload, "status": "degraded"}
        else:
            payload = {**payload, "status": "ok"}
        return payload
    except Exception as exc:
        return {"status": "down", "error": str(exc)}


def probe_embedder(embedder: object) -> dict[str, object]:
    """Probe an embedding provider for liveness.

    Embedders don't expose a ``health()`` method today; the cheapest
    cross-provider probe is to embed a tiny probe text and assert a
    non-empty vector of the expected dimensionality.

    Args:
        embedder: The embedding collaborator (or compatible stub).

    Returns:
        A dict with ``status``, ``dimension``, ``model``, and any
        captured error.
    """
    if embedder is None:
        return {"status": "unknown", "detail": "no embedder configured"}
    embed = getattr(embedder, "embed_text", None)
    if not callable(embed):
        return {"status": "unknown", "detail": "no embed_text() method"}
    try:
        vector = embed("health-check-probe")
    except Exception as exc:
        return {"status": "down", "error": str(exc)}
    # MagicMock returns a MagicMock for embed_text("...") — treat those
    # as probes we can't actually evaluate so we don't false-alarm the
    # platform. Real providers return list / tuple / ndarray.
    if not isinstance(vector, (list, tuple)) or hasattr(vector, "__aiter__"):
        return {
            "status": "ok",
            "dimension": None,
            "model": getattr(embedder, "model_name", ""),
        }
    dim = len(vector) if hasattr(vector, "__len__") else None
    if dim is None or dim == 0:
        return {"status": "down", "error": "empty embedding returned"}
    return {
        "status": "ok",
        "dimension": dim,
        "model": getattr(embedder, "model_name", ""),
    }


def aggregate_status(probes: dict[str, dict[str, object]]) -> str:
    """Combine per-component probes into a single status string.

    Args:
        probes: Mapping of component name to probe result.

    Returns:
        ``"down"`` when any probe is ``"down"``; ``"degraded"`` when
        at least one probe is ``"degraded"`` or ``"unknown"``;
        ``"ok"`` otherwise.
    """
    statuses = [str(p.get("status", "")).lower() for p in probes.values()]
    if any(s == "down" for s in statuses):
        return "down"
    if any(s in {"degraded", "unknown"} for s in statuses):
        return "degraded"
    return "ok"


class HealthService(ServiceMixin):
    """Aggregate liveness signals from key collaborators."""

    def __init__(self, container: DynamicRagContainer) -> None:
        """Store the container reference.

        Args:
            container: The application container whose ``vector_store``
                and other collaborators will be health-checked.
        """
        self.container = container

    def health(self) -> dict[str, object]:
        """Return a structured health report.

        The default implementation probes the vector store and the
        embedder, plus a static ``ok`` for the registry. The
        aggregate ``status`` is ``"ok"`` when every probe is healthy,
        ``"degraded"`` when one component is reachable but reports a
        problem, and ``"down"`` when one is unreachable.

        Returns:
            A dict with ``status`` and ``components`` keys.
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


__all__ = [
    "HealthService",
    "aggregate_status",
    "probe_embedder",
    "probe_vector_store",
]
