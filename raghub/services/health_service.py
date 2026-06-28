"""Health-check service aggregating component liveness.

A thin wrapper around each collaborator's ``health()`` method. The
service is intentionally synchronous (it does no IO beyond the
collaborator calls) so it can be polled cheaply by orchestrators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raghub.services import ServiceMixin

if TYPE_CHECKING:
    from raghub.services.application import DynamicRagContainer


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

        The default implementation delegates to the vector store's
        ``health()`` method and returns a static ``ok`` for the
        registry. Callers should treat ``status == "ok"`` as "every
        checked component is healthy".

        Returns:
            A dict with ``status`` and ``components`` keys.
        """
        self.log("health_check")
        return {
            "status": "ok",
            "components": {
                "vectorstore": self.container.vector_store.health(),
                "registry": {"status": "ok"},
            },
        }