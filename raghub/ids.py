"""Value-object IDs used across the framework.

Per AGENTS.md §1755-1781 (primitive obsession), repeated string IDs
should be wrapped in nominal types so call sites communicate intent.
Using :func:`typing.NewType` avoids any runtime cost while giving
mypy / pyright a discriminator.

Usage::

    from raghub.ids import TenantId, UserId

    def get_documents(tenant: TenantId, user: UserId) -> list[Document]:
        ...
"""

from __future__ import annotations

from typing import NewType

TenantId = NewType("TenantId", str)
UserId = NewType("UserId", str)
DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)
SessionId = NewType("SessionId", str)
JobId = NewType("JobId", str)

__all__ = [
    "TenantId",
    "UserId",
    "DocumentId",
    "ChunkId",
    "SessionId",
    "JobId",
]