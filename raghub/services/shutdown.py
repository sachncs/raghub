"""Shutdown coordinator: release collaborators held by the container."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raghub.services.container import RagContainer


class Shutdown:
    """Release collaborators held by the :class:`RagContainer`."""

    SHUTDOWN_TARGETS: tuple[str, ...] = (
        "background_ingestion",
        "ingestion",
        "image_store",
        "vector_store",
        "store",
        "uow",
    )

    def __init__(self, container: "RagContainer") -> None:
        """Store the container reference."""
        self.container = container

    async def release(self) -> None:
        """Close every owned collaborator in order."""
        for attr in self.SHUTDOWN_TARGETS:
            collaborator = getattr(self.container, attr, None)
            if collaborator is None:
                continue
            close = getattr(collaborator, "close", None) or getattr(collaborator, "shutdown", None)
            if close is None:
                continue
            result = close()
            if asyncio.iscoroutine(result):
                await result


__all__ = ["Shutdown"]
