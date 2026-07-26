"""Document ingestion workflows."""

from typing import Any

from .service import DocumentIngestionService, IngestionResult


def __getattr__(name: str) -> Any:
    """Resolve ingestion service exports."""
    if name == "DocumentIngestionService":
        return DocumentIngestionService
    if name == "IngestionResult":
        return IngestionResult
    raise AttributeError(f"module 'raghub.ingestion' has no attribute {name!r}")


__all__ = ["DocumentIngestionService", "IngestionResult"]
