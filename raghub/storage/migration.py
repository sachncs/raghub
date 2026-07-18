"""One-shot migration utility from the JSON stores to the SQLite stores.

This module provides :func:`migrate_from_json`, a coroutine that reads
documents and sessions from the legacy JSON-backed stores and writes
them into the SQLite-backed stores. It exists to support deployments
that started on the JSON store and want to move to the more durable
SQLite backend without losing history.

The migration is **additive**: it does not delete or modify the source
JSON files. Run it once, verify the SQLite data, then archive or
remove the JSON files manually.

The JSON document registry stores every version of a document; the
SQLite store persists them all via composite key ``(document_id, version)``.
The JSON session store is keyed by session id (not token); we insert
each one explicitly so the session_id, token, expires_at, and history
survive the move.
"""

from __future__ import annotations

from pathlib import Path


async def migrate_from_json(
    db_path: str | Path,
    registry_path: str | Path,
    sessions_path: str | Path,
) -> None:
    from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
    from raghub.repositories.sqlite_session_repo import SqliteSessionRepository
    from raghub.storage.json_registry import JsonDocumentRegistry
    from raghub.storage.session_store import JsonSessionStore

    registry = SqliteDocumentRepository(db_path)
    await registry.initialize()

    json_registry = JsonDocumentRegistry(Path(registry_path))
    for versions in json_registry.documents.values():
        for doc in versions:
            await registry.save(doc)

    session_repo = SqliteSessionRepository(db_path)
    await session_repo.initialize()

    json_sessions = JsonSessionStore(Path(sessions_path), timeout_seconds=3600)
    for session in json_sessions.sessions.values():
        await session_repo.create(session)
