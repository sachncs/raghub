"""Document ingestion workflows.

The legacy :class:`DocumentIngestionService` and
:class:`IngestionResult` are imported lazily to keep the base
package free of ``aiosqlite`` at import time.

The legacy service is now a **thin wrapper** around
:class:`raghub.pipelines.rag.IngestPipeline`. All real work
(conversion, chunking, embedding, indexing, deduplication) lives
in the canonical pipeline; the legacy class preserves the public
method surface for FastAPI, CLI, streamlit, and background jobs.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .service import DocumentIngestionService, IngestionResult


def __getattr__(name: str) -> Any:
    """Lazily expose ingestion services."""
    if name in {"DocumentIngestionService", "IngestionResult"}:
        from . import service as _service

        return getattr(_service, name)
    raise AttributeError(f"module 'raghub.ingestion' has no attribute {name!r}")


__all__ = ["DocumentIngestionService", "IngestionResult"]
