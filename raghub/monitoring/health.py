"""Health-check service that aggregates component status.

The health service iterates over a dict of named components and
invokes each one's ``health()`` method (when available). Components
that fail are recorded as ``{"status": "error", "error": str(exc)}``
and the overall response is degraded. This pattern makes it easy to
add new components (vector store, database, LLM client) without
touching the health surface.
"""

from __future__ import annotations

from typing import Any


class HealthService:
    """Aggregate ``health()`` calls across a set of named components."""

    def __init__(self, components: dict[str, Any]) -> None:
        """Initialise the service.

        Args:
            components: A mapping of component name → component.
                Each component should expose a ``health() -> dict``
                method; components without one are reported as
                ``{"status": "ok"}``.
        """
        self.components = components

    def check(self) -> dict[str, Any]:
        """Aggregate component health into a single status dict.

        Returns:
            A dict with two keys:

            * ``status``: ``"ok"`` when every component is healthy,
              ``"degraded"`` otherwise.
            * ``components``: per-component status dictionaries keyed
              by component name.

        Note:
            Any exception raised by a component's ``health()`` is
            caught and reported as ``{"status": "error", "error": str(exc)}``
            so a single broken component cannot take down the whole
            health endpoint.
        """
        statuses: dict[str, Any] = {}
        healthy = True
        for name, component in self.components.items():
            try:
                # Components without a ``health()`` method are treated
                # as trivially healthy; this lets the service accept
                # plain dicts or simple stub objects in tests.
                statuses[name] = (
                    component.health() if hasattr(component, "health") else {"status": "ok"}
                )
            except Exception as exc:  # pragma: no cover - defensive path
                statuses[name] = {"status": "error", "error": str(exc)}
                healthy = False
        return {"status": "ok" if healthy else "degraded", "components": statuses}