"""Legacy domain models — retained for backward compatibility.

These active-record-style models predate the canonical Pydantic
models in :mod:`raghub.models`. New code should use the models
from :mod:`raghub.models.canonical` instead.
"""

from .document import Document
from .chunk import Chunk
from .session import Session
from .repositories import DocumentRepository, ChunkRepository, SessionRepository, UnitOfWork

__all__ = [
    "Chunk",
    "ChunkRepository",
    "Document",
    "DocumentRepository",
    "Session",
    "SessionRepository",
    "UnitOfWork",
]
