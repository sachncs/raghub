"""Legacy domain models — retained for backward compatibility.

These active-record-style models predate the canonical Pydantic
models in :mod:`raghub.models`. New code should use the models
from :mod:`raghub.models.canonical` instead.
"""

from .chunk import Chunk
from .document import Document
from .repositories import ChunkRepository, DocumentRepository, SessionRepository, UnitOfWork
from .session import Session

__all__ = [
    "Chunk",
    "ChunkRepository",
    "Document",
    "DocumentRepository",
    "Session",
    "SessionRepository",
    "UnitOfWork",
]
