"""Pipeline helpers — small, dependency-light utilities.

Co-locates the timer, awaitable bridge, filter canonicalisation,
checksum helper, and chunk materialiser. None of these depend on
the heavy collaborators (embedder, vector store, generator), so
the other pipeline submodules can import from here without
creating cycles.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
from types import TracebackType
from typing import Any

from raghub.models import Bundle, Chunk, Classification, deterministic_id


def awaitable(value: Any) -> Any:
    """Make ``await`` work for either sync return values or coroutines.

    Lifts a sync result into an inline coroutine so the query
    pipeline can drive both the async and the sync ``generate``
    path through the same call site.
    """
    if inspect.isawaitable(value):
        return value

    # Inline async coroutine factory (one level deep, no nested def).
    async def lift() -> Any:
        """Lift a sync return value into a coroutine."""
        await asyncio.sleep(0)
        return value

    return lift()


class DurationTimer(AbstractContextManager["DurationTimer"]):
    """Set ``context.metadata["duration_ms"]`` on exit."""

    def __init__(self, context: Any) -> None:
        """Store the context; the start time is captured on entry."""
        self.context = context
        self.start: float = 0.0

    def __enter__(self) -> DurationTimer:
        """Capture the start time and return ``self`` for ``as`` binding."""
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Record the elapsed milliseconds in ``context.metadata``."""
        self.context.metadata["duration_ms"] = (time.perf_counter() - self.start) * 1000.0


def canonical_filters(filters: dict[str, Any] | str | None) -> tuple[tuple[str, Any], ...]:
    """Flatten ``filters`` into a hashable tuple."""
    if filters is None:
        return ()
    if isinstance(filters, str):
        return (("raw", filters),)
    items = []
    for key, value in sorted(filters.items()):
        if isinstance(value, list):
            value = tuple(value)
        items.append((key, value))
    return tuple(items)


def get_chunks(bundle: Bundle, document_id: str, company: str = "") -> list[Chunk]:
    """Materialise the :class:`Chunk` list for a bundle's sections."""
    from raghub.tenants import current

    chunks: list[Chunk] = []
    tenant_company = company or bundle.metadata.get("company", "")
    ctx = current()
    tenant_id = ctx.tenant_id if ctx else ""
    for section in bundle.sections:
        for block in section.blocks:
            if block.kind.value != "text":
                continue
            text = (block.content or "").strip()
            if not text:
                continue
            chunk_id = deterministic_id(
                "chunk",
                document_id,
                str(section.index),
                block.block_id,
                text[:64],
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version=1,
                    page=(section.page_numbers[0] if section.page_numbers else section.index),
                    source_location=section.source_location or bundle.source_uri,
                    section=section.heading,
                    company=tenant_company,
                    owner=bundle.metadata.get("owner", ""),
                    department=bundle.metadata.get("department", ""),
                    tenant_id=tenant_id,
                    text=text,
                    checksum=sha256(text.encode("utf-8")).hexdigest(),
                    metadata={
                        "block_kind": "text",
                        "block_id": block.block_id,
                        "section_index": section.index,
                    },
                )
            )
    return chunks


def sha256_checksum(file_bytes: bytes) -> str:
    """SHA-256 of the raw file content."""
    return sha256(file_bytes).hexdigest()


def primary_company(user: Any) -> str:
    """Return the primary company for a :class:`User`."""
    if user is None:
        return ""
    companies = getattr(user, "allowed_companies", None) or []
    if getattr(user, "is_admin", False):
        return ""
    if not companies:
        return ""
    return str(companies[0])


@dataclass(frozen=True, slots=True)
class IngestResolvedMetadata:
    """Resolved per-request metadata for :meth:`Ingest.run`."""

    normalized_metadata: dict[str, Any]
    document_id: str
    version: int
    tenant_company: str
    owner: str
    classification: Classification
    mime_type: str
    language: str


@dataclass(frozen=True, slots=True)
class QueryContext:
    """Per-request context passed through the :class:`QueryPipeline` helpers."""

    question: str
    top_k: int
    user_filter: dict[str, Any] | str
    user: Any | None
    session_id: str | None
    response_model: Any
    record: bool
    history: list[Any]
    rbac_filter: dict[str, Any] | str
    user_id: str | None
    scope: tuple[bool, tuple[str, ...], tuple[str, ...]]


__all__ = [
    "DurationTimer",
    "IngestResolvedMetadata",
    "QueryContext",
    "awaitable",
    "canonical_filters",
    "get_chunks",
    "primary_company",
    "sha256_checksum",
]
