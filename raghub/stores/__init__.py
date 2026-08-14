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
"""

from raghub.stores.connection import Database, JsonSessions, Sessions
from raghub.stores.documents import Documents, Snapshot
from raghub.stores.images import ImageStore
from raghub.stores.migrate import migrate_from_json
from raghub.stores.schema import SQLITE_SCHEMA

__all__ = [
    "SQLITE_SCHEMA",
    "Database",
    "Documents",
    "ImageStore",
    "JsonSessions",
    "Sessions",
    "Snapshot",
    "migrate_from_json",
]
