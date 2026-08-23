"""One-shot JSON -> SQLite migration utility.

Reads documents and sessions from the JSON-backed stores
and writes them into the SQLite-backed stores.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from raghub.constants import DEFAULT_SESSION_TIMEOUT_SECONDS
from raghub.repos import DocStore, SessionStore
from raghub.stores.connection import Sessions
from raghub.stores import documents as _documents

__all__ = ["migrate_from_json"]


async def migrate_from_json(
    db_path: str | Path,
    registry_path: str | Path,
    sessions_path: str | Path,
    *,
    show_progress: bool = True,
) -> None:
    """Migrate documents and sessions from JSON to SQLite.

    Reads documents and sessions from the JSON-backed stores
    and writes them into the SQLite-backed stores.

    Args:
        db_path: Path to the SQLite registry db (created if missing).
        registry_path: Path to the source JSON registry file.
        sessions_path: Path to the source JSON sessions file.
        show_progress: Wrap each step in a :class:`tqdm.tqdm` bar.

    """
    registry = DocStore(db_path)
    await registry.initialize()

    documents = _documents.Documents(Path(registry_path))
    all_versions = [doc for versions in documents.documents.values() for doc in versions]
    for doc in tqdm(
        all_versions,
        desc="Migrating documents",
        disable=not show_progress,
        unit="doc",
    ):
        await registry.save(doc)

    session_repo = SessionStore(db_path)
    await session_repo.initialize()

    json_sessions = Sessions.json(
        Path(sessions_path), timeout_seconds=DEFAULT_SESSION_TIMEOUT_SECONDS
    )
    for session in tqdm(
        list(json_sessions.sessions.values()),
        desc="Migrating sessions",
        disable=not show_progress,
        unit="session",
    ):
        await session_repo.create_from_record(session)
