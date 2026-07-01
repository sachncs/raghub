"""Transport schemas for the FastAPI reference app.

This module re-exports the canonical Pydantic request/response models
defined in :mod:`raghub.models.api`. Keeping the re-exports here lets
route modules do ``from raghub.api.schemas import QueryResponse`` while
the schema definitions stay co-located with the domain models.
"""

from __future__ import annotations

from raghub.models.api import (
    AuthLoginRequest,
    AuthLoginResponse,
    BatchIngestResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)

__all__ = [
    "AuthLoginRequest",
    "AuthLoginResponse",
    "BatchIngestResponse",
    "DocumentUploadResponse",
    "QueryRequest",
    "QueryResponse",
]