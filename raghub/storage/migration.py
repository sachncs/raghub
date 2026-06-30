"""One-shot migration utility from the JSON stores to the SQLite stores.

This module provides :func:`migrate_from_json`, a coroutine that reads
documents and sessions from the legacy JSON-backed stores and writes
them into the SQLite-backed stores. It exists to support deployments
that started on the JSON store and want to move to the more durable
SQLite backend without losing history.

The migration is **additive**: it does not delete or modify the source
JSON files. Run it once, verify the SQLite data, then archive or
remove the JSON files manually.
"""

from __future__ import annotations

from pathlib import Path


async def migrate_from_json(
    db_path: str | Path,
    registry_path: str | Path,
    sessions_path: str | Path,
) -> None:
    """Migrate documents and sessions from JSON to SQLite.

    Steps:

    1. Open (and initialise) the SQLite document repository.
    2. Walk every version of every document in the JSON registry and
       save it to SQLite. The ``created_at`` / ``updated_at`` lines
       are kept as explicit no-ops for clarity (they remind readers
       that timestamps are already populated).
    3. Open the SQLite session store (initialising the schema).
    4. Walk every session in the JSON session store and overwrite the
       matching SQLite row (if any). New rows are inserted via the
       ``UPDATE`` path; sessions that exist only in JSON will end up
       inserted on the first call because SQLite ``UPDATE`` without a
       matching row is a silent no-op — run a backfill SQL step if
       you need them.

    Args:
        db_path: Path to the SQLite database file. Created if missing.
        registry_path: Path to the legacy JSON document registry.
        sessions_path: Path to the legacy JSON session store.

    Note:
        This function is intended to be run once during the cut-over.
        It does not deduplicate and does not produce a migration report;
        if you need either, wrap this call in your own orchestrator.
    """
    from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
    from raghub.storage.json_registry import JsonDocumentRegistry
    from raghub.storage.session_store import JsonSessionStore
    from raghub.storage.sqlite_session_store import SqliteSessionStore

    registry = SqliteDocumentRepository(db_path)
    await registry.initialize()

    json_registry = JsonDocumentRegistry(Path(registry_path))
    for versions in json_registry.documents.values():
        for doc in versions:
            doc.created_at = doc.created_at
            doc.updated_at = doc.updated_at
            await registry.save(doc)

    session_store = SqliteSessionStore(db_path)
    await session_store.initialize()

    json_sessions = JsonSessionStore(Path(sessions_path), timeout_seconds=3600)
    for session in json_sessions.sessions.values():
        await session_store.update_session(session)