"""Durable storage adapters.

Re-exports every public symbol from the focused per-feature submodules
so existing imports (``from raghub.stores import Database``,
``from raghub.stores import Sessions``) keep working without modification.

The package layout:

- :mod:`raghub.stores.connection` - Database, JsonSessions, Sessions
- :mod:`raghub.stores.documents` - Documents, Snapshot
- :mod:`raghub.stores.images` - ImageStore
- :mod:`raghub.stores.migrate` - migrate_from_json utility
- :mod:`raghub.stores.schema` - SQLITE_SCHEMA constant
- :mod:`raghub.stores.pgvector` - Postgres/pgvector backend
- :mod:`raghub.stores.vector_base` - vector store base + helpers
- :mod:`raghub.stores.vector_memory` - in-process memory store
- :mod:`raghub.stores.vector_sqlite` - SQLite-backed vector store
- :mod:`raghub.stores.vector_factory` - build_store() factory
"""

from raghub.stores.connection import Database, JsonSessions, Sessions
from raghub.stores.documents import Documents, Snapshot
from raghub.stores.images import ImageStore
from raghub.stores.migrate import migrate_from_json
from raghub.stores.schema import SQLITE_SCHEMA
from raghub.stores.vector_base import (
    MemoryVectorRecord,
    Store,
    matches_metadata_dict,
    matches_metadata_string,
)
from raghub.stores.vector_factory import build_store
from raghub.stores.vector_memory import MemoryStore
from raghub.stores.vector_sqlite import SqliteStore

__all__ = [
    "SQLITE_SCHEMA",
    "Database",
    "Documents",
    "ImageStore",
    "JsonSessions",
    "MemoryStore",
    "MemoryVectorRecord",
    "Sessions",
    "Snapshot",
    "SqliteStore",
    "Store",
    "build_store",
    "matches_metadata_dict",
    "matches_metadata_string",
    "migrate_from_json",
]
