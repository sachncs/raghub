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

from tqdm import tqdm

import raghub.repositories as repositories
import raghub.storage.json_registry as json_registry
import raghub.storage.session_store as session_store


async def migrate_from_json(
    db_path: str | Path,
    registry_path: str | Path,
    sessions_path: str | Path,
    *,
    show_progress: bool = True,
) -> None:
    """Migrate documents and sessions from JSON to SQLite.

    Args:
        db_path: Path to the SQLite registry db (created if missing).
        registry_path: Path to the source JSON registry file.
        sessions_path: Path to the source JSON session file.
        show_progress: When ``True`` (default), wrap each step in a
            :class:`tqdm.tqdm` progress bar.

    Returns:
        ``None``.

    Raises:
        Any exception raised by the underlying :class:`SqliteDocumentRepository`
        or :class:`SqliteSessionRepository` propagates to the caller.
    """
    registry = repositories.SqliteDocumentRepository(db_path)
    await registry.initialize()

    json_registry_instance = json_registry.JsonDocumentRegistry(Path(registry_path))
    all_versions = [doc for versions in json_registry_instance.documents.values() for doc in versions]
    for doc in tqdm(
        all_versions, desc="Migrating documents", disable=not show_progress, unit="doc"
    ):
        await registry.save(doc)

    session_repo = repositories.SqliteSessionRepository(db_path)
    await session_repo.initialize()

    json_sessions = session_store.JsonSessionStore(Path(sessions_path), timeout_seconds=3600)
    for session in tqdm(
        list(json_sessions.sessions.values()),
        desc="Migrating sessions",
        disable=not show_progress,
        unit="session",
    ):
        await session_repo.create_from_record(session)
