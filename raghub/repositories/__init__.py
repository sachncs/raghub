"""SQLite-backed repository implementations.

Concrete implementations of the legacy repository protocols
defined in :mod:`raghub.domain.repositories`. These are retained
for backward compatibility; new code should use the higher-level
interfaces in :mod:`raghub.interfaces` and :mod:`raghub.knowledge`.
"""

from .sqlite_chunk_repo import SqliteChunkRepository
from .sqlite_document_repo import SqliteDocumentRepository
from .sqlite_session_repo import SqliteSessionRepository
from .unit_of_work import UnitOfWork

__all__ = [
    "SqliteChunkRepository",
    "SqliteDocumentRepository",
    "SqliteSessionRepository",
    "UnitOfWork",
]
