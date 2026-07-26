"""Resource shutdown coordinator.

Walks the well-known collaborator list on the
:class:`DynamicRagContainer` and closes each in order. Failures
propagate to the caller — the legacy per-collaborator ``try`` /
``except`` has been removed because it swallowed every shutdown
error and hid misconfigured resources. Callers (the FastAPI
lifespan, CLI teardown) can now decide how to handle a stuck
collaborator.
"""

from __future__ import annotations

import asyncio
from typing import Any


_SHUTDOWN_TARGETS: tuple[str, ...] = (
    "background_ingestion",
    "ingestion",
    "image_store",
    "vector_store",
    "store",
    "uow",
)


class ShutdownCoordinator:
    """Release collaborators held by the :class:`DynamicRagContainer`.

    The coordinator is intentionally stateless — every collaborator
    owns the resource it backs. The container is the single source
    of truth, so the coordinator just iterates the well-known list.

    Attributes:
        container: The application container whose collaborators
            will be released.
    """

    def __init__(self, container: Any) -> None:
        """Store the container reference."""
        self.container = container

    async def release(self) -> None:
        """Close every owned collaborator in order.

        Each collaborator is closed via ``close()`` or ``shutdown()``
        when present. The first failing close propagates to the
        caller; subsequent collaborators are not closed. This is
        intentional — silently swallowing per-collaborator failures
        hides misconfiguration, and the lifespan caller is in a
        better position to decide whether to retry or abort.
        """
        for attr in _SHUTDOWN_TARGETS:
            collaborator = getattr(self.container, attr, None)
            if collaborator is None:
                continue
            close = getattr(collaborator, "close", None) or getattr(collaborator, "shutdown", None)
            if close is None:
                continue
            result = close()
            if asyncio.iscoroutine(result):
                await result


__all__ = ["ShutdownCoordinator"]