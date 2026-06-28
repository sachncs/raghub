from __future__ import annotations

from pathlib import Path

from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
from raghub.storage.json_registry import JsonDocumentRegistry
from raghub.storage.session_store import JsonSessionStore
from raghub.storage.sqlite_session_store import SqliteSessionStore


async def migrate_from_json(db_path: str | Path, registry_path: str | Path, sessions_path: str | Path) -> None:
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
