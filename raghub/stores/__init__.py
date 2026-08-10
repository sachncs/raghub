"""Durable storage adapters.

Re-exports every public symbol from the focused per-feature submodules
so existing imports (``from raghub.stores import Database``,
``from raghub.stores import Sessions``) keep working without modification.

The package internal layout is:
- :mod:`raghub.stores._conn` - Database, JsonSessions, Sessions, helpers
- :mod:`raghub.stores._images` - ImageStore
- :mod:`raghub.stores._documents` - Documents, Snapshot
- :mod:`raghub.stores._schema` - SQLITE_SCHEMA
- :mod:`raghub.stores._migrate` - migrate_from_json utility
- :mod:`raghub.stores.pgvector` - Postgres/pgvector backend (unchanged)

Names follow the no-suffix rule:
``Database`` (was ``Database``), ``ImageStore`` (was ``FilesystemImageStore``),
``Documents`` (was ``JsonDocumentRegistry``), ``Sessions`` (was ``SqliteSessionStore``,
the canonical SQLite-backed class), ``JsonSessions`` (was ``JsonSessionStore``).
Call ``Sessions.json(path, timeout)`` to get a JSON-backed instance.
"""

from raghub.stores._conn import (
    Database,
    JsonSessions,
    Sessions,
    __keyed,
    serialize_overrides,
)
from raghub.stores._documents import Documents, Snapshot
from raghub.stores._images import ImageStore
from raghub.stores._migrate import migrate_from_json
from raghub.stores._schema import SQLITE_SCHEMA

__all__ = [
    "SQLITE_SCHEMA",
    "Database",
    "Documents",
    "ImageStore",
    "JsonSessions",
    "Sessions",
    "Snapshot",
    "__keyed",
    "migrate_from_json",
    "serialize_overrides",
]
