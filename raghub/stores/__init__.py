"""Durable storage adapters.

Defines the durable storage adapters in one helper
file. Class summary::

    Database                 - shared :class:`aiosqlite` connection manager.
    ImageStore               - content-addressable image storage on disk.
    Documents                - JSON-backed document registry with version history.
    Snapshot                 - in-memory snapshot of :class:`Documents`.
    JsonSessions            - JSON-backed session store. Reach for the canonical
                              :class:`Sessions` and call ``Sessions.json(...)``;
                              direct use of :class:`JsonSessions` is fine but
                              the factory is the preferred entry point.
    Sessions                - SQLite-backed session store (production default).

Module-level helpers::

    SQLITE_SCHEMA            - DDL for ``documents``, ``chunks``, ``sessions``, ``users``.
    migrate_from_json        - one-shot JSON → SQLite migration utility.

Names follow the no-suffix rule:
``Database`` (was ``Database``), ``ImageStore`` (was ``FilesystemImageStore``),
``Documents`` (was ``JsonDocumentRegistry``), ``Sessions`` (was ``SqliteSessionStore``,
the canonical SQLite-backed class), ``JsonSessions`` (was ``JsonSessionStore``).
Call ``Sessions.json(path, timeout)`` to get a JSON-backed instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from tqdm import tqdm

from raghub.await_sync import capture
from raghub.domain import Database
from raghub.errors import AuthenticationError, MissingDepError, RagHubError
from raghub.io import atomic_write_json, load_json
from raghub.models import (
    Document,
    DocumentLifecycleStatus,
    Session,
    Turn,
)


def __keyed(row: Any) -> bool:
    """Return ``True`` when ``row`` is an :class:`aiosqlite.Row`.

    Defers the import so the module loads without ``aiosqlite``.
    """
    try:
        import aiosqlite
    except ImportError:
        return False
    return isinstance(row, aiosqlite.Row)


def serialize_overrides(overrides: dict[str, Any] | None) -> str:
    """Serialize a session's overrides mapping for SQLite persistence.

    Args:
        overrides: The mapping; ``None`` and empty mappings both
            serialise to the JSON object string ``"{}"``.

    Returns:
        A JSON string suitable for an SQLite TEXT column.

    """
    return json.dumps(overrides or {})


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    owner TEXT NOT NULL,
    organization TEXT NOT NULL,
    department TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    classification TEXT DEFAULT 'internal',
    visibility TEXT DEFAULT 'organization',
    status TEXT DEFAULT 'NEW',
    filename TEXT DEFAULT '',
    file_type TEXT DEFAULT '',
    mime_type TEXT DEFAULT '',
    chunk_count INTEGER DEFAULT 0,
    chunk_ids TEXT DEFAULT '[]',
    error TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    page INTEGER DEFAULT 0,
    source_location TEXT DEFAULT '',
    section TEXT DEFAULT '',
    company TEXT NOT NULL,
    owner TEXT NOT NULL,
    department TEXT DEFAULT '',
    classification TEXT DEFAULT 'internal',
    created_at TEXT NOT NULL,
    embedding_model TEXT DEFAULT '',
    hash TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    history TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    allowed_companies TEXT DEFAULT '[]',
    allowed_groups TEXT DEFAULT '[]',
    is_admin INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Shared SQLite connection manager
# ---------------------------------------------------------------------------


class Database:
    """Manages a shared :class:`aiosqlite.Connection` with WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        """Store the db path; lazy-connect on first :meth:`connect`."""
        self.db_path = str(db_path)
        self.conn: Any | None = None

    async def connect(self) -> Any:
        """Open (or reuse) the underlying aiosqlite connection."""
        if self.conn is None:
            try:
                import aiosqlite
            except ImportError:
                raise MissingDepError(
                    "aiosqlite",
                    "pip install raghub[auth]",
                ) from None
            self.conn = await aiosqlite.connect(self.db_path, isolation_level=None)
            self.conn.row_factory = aiosqlite.Row
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
            await self.conn.execute("PRAGMA foreign_keys=ON")
        return self.conn

    async def close(self) -> None:
        """Checkpoint the WAL and close the connection. Idempotent."""
        if self.conn is not None:
            conn = self.conn
            await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await conn.close()
            self.conn = None

    @property
    def connection(self) -> Any:
        """Return the live connection or raise if not yet connected."""
        if self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn


# ---------------------------------------------------------------------------
# Image store (content-addressable filesystem)
# ---------------------------------------------------------------------------


class ImageStore:
    """Content-addressable image storage on the local filesystem.

    Images are stored by their SHA-256 content hash under
    ``<base_path>/<hash[:2]>/<hash><extension>``. The two-character prefix
    subdirectory keeps any single directory from growing unboundedly.
    """

    def __init__(self, base_path: str | Path = "./data/images") -> None:
        """Store the root directory; created lazily on first :meth:`save`."""
        self.base_path = Path(base_path)

    def save(self, file_bytes: bytes, extension: str = ".png") -> str:
        """Persist ``file_bytes`` and return the content hash."""
        content_hash = sha256(file_bytes).hexdigest()
        subdir = content_hash[:2]
        dest_dir = self.base_path / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{content_hash}{extension}"
        if not dest_path.exists():
            dest_path.write_bytes(file_bytes)
        return content_hash

    def get_path(self, content_hash: str, extension: str = ".png") -> Path | None:
        """Resolve a content hash to its filesystem path, or ``None``."""
        path = self.base_path / content_hash[:2] / f"{content_hash}{extension}"
        return path if path.exists() else None

    def get_bytes(self, content_hash: str, extension: str = ".png") -> bytes | None:
        """Return the raw bytes for ``content_hash``, or ``None``."""
        path = self.get_path(content_hash, extension)
        return path.read_bytes() if path is not None else None

    def delete(self, content_hash: str, extension: str = ".png") -> bool:
        """Delete the file for ``content_hash``. Returns whether a file was removed."""
        path = self.get_path(content_hash, extension)
        if path is not None:
            path.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# JSON-backed document registry
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """In-memory snapshot of a document registry."""

    documents: dict[str, list[Document]]
    checksum_index: dict[str, tuple[str, int]]


class Documents:
    """JSON-backed persistent registry for versioned documents.

    Versions are stored in append order; each new version that exceeds
    the previous latest's number automatically archives the prior latest
    (see :meth:`save_version`).
    """

    def __init__(self, path: Path) -> None:
        """Initialise the registry and load the existing JSON state."""
        self.path = path
        self.lock = RLock()
        self.documents: dict[str, list[Document]] = {}
        self.checksum_index: dict[str, tuple[str, int]] = {}
        self.load()

    def load(self) -> None:
        """Hydrate in-memory state from disk.

        Tolerates a missing or malformed file by resetting to empty
        state; this is the behaviour we want for first-run startup.
        """
        if self.path.exists() and not self.path.read_text(encoding="utf-8").lstrip().startswith(
            "{"
        ):
            self.documents = {}
            self.checksum_index = {}
            return
        payload = load_json(self.path, default={"documents": {}, "checksum_index": {}})
        documents = payload.get("documents", {})
        checksum_index = payload.get("checksum_index", {})
        self.documents = {
            document_id: [Document.model_validate(item) for item in versions]
            for document_id, versions in documents.items()
            if isinstance(versions, list)
        }
        self.checksum_index = {
            checksum: tuple(value)
            for checksum, value in checksum_index.items()
            if isinstance(value, list)
        }

    def save(self) -> None:
        """Persist in-memory state to disk atomically.

        Raises:
            RagHubError: If the atomic write fails.

        """
        _, error = capture(
            atomic_write_json,
            self.path,
            {
                "documents": {
                    document_id: [version.model_dump(mode="json") for version in versions]
                    for document_id, versions in self.documents.items()
                },
                "checksum_index": {
                    checksum: list(value) for checksum, value in self.checksum_index.items()
                },
            },
        )
        if error is not None:
            raise RagHubError(str(error)) from error

    def save_version(self, document: Document) -> Document:
        """Persist a new or updated :class:`Document`."""
        with self.lock:
            versions = self.documents.setdefault(document.id, [])
            for index, existing in enumerate(versions):
                if existing.version == document.version:
                    # Replace-in-place: an out-of-order write for an
                    # existing version number should update, not append.
                    versions[index] = document
                    break
            else:
                # ``for/else`` runs when the loop completes without a
                # ``break`` — a brand-new version number.
                if versions and document.version > versions[-1].version:
                    versions[-1].status = DocumentLifecycleStatus.ARCHIVED
                    versions[-1].updated_at = datetime.now(UTC)
                versions.append(document)
            self.checksum_index[document.checksum] = (
                document.id,
                document.version,
            )
            self.save()
            return document

    def get_latest(self, document_id: str) -> Document | None:
        """Return the highest-versioned entry for ``document_id``."""
        with self.lock:
            versions = self.documents.get(document_id, [])
            if not versions:
                return None
            return max(versions, key=lambda v: v.version)

    def get_specific_version(self, document_id: str, version: int) -> Document | None:
        """Return a specific historical version, or ``None``."""
        with self.lock:
            for item in self.documents.get(document_id, []):
                if item.version == version:
                    return item
            return None

    def get_by_checksum(self, checksum: str) -> Document | None:
        """Look up the document owning ``checksum``."""
        with self.lock:
            locator = self.checksum_index.get(checksum)
            if locator is None:
                return None
            return self.get_specific_version(locator[0], locator[1])

    def list_accessible(self, companies: list[str]) -> list[Document]:
        """Return the latest version of every non-archived document."""
        with self.lock:
            result: list[Document] = []
            for versions in self.documents.values():
                latest = versions[-1]
                if (
                    latest.organization in companies
                    and latest.status != DocumentLifecycleStatus.ARCHIVED
                ):
                    result.append(latest)
            return result

    def archive(self, document_id: str) -> None:
        """Archive the latest version of ``document_id``. No-op if unknown."""
        with self.lock:
            latest = self.get_latest(document_id)
            if latest is None:
                return
            latest.status = DocumentLifecycleStatus.ARCHIVED
            latest.updated_at = datetime.now(UTC)
            self.save()

    def dump(self) -> Snapshot:
        """Return an in-memory snapshot of the registry."""
        with self.lock:
            return Snapshot(self.documents, self.checksum_index)


# ---------------------------------------------------------------------------
# Session stores
# ---------------------------------------------------------------------------


class JsonSessions:
    """JSON-backed session store with sliding-window inactivity expiry.

    Every successful resolve resets the inactivity timer (expiry bumped
    to ``now + timeout``), implementing the classic "sliding session"
    behaviour: an active user is never logged out, idle sessions are
    pruned on access.

    Reach for the canonical :class:`Sessions` class and call
    ``Sessions.json(...)`` rather than constructing this directly; the
    factory keeps the backends interchangeable.
    """

    def __init__(self, path: str | Path, timeout_seconds: int) -> None:
        """Initialise the store and load existing state."""
        self.path = Path(path)
        self.timeout = timedelta(seconds=timeout_seconds)
        self.lock = RLock()
        self.sessions: dict[str, Session] = {}
        self.load()

    def load(self) -> None:
        """Hydrate in-memory state from disk.

        Tolerates a missing or malformed file by resetting to empty.
        """
        if self.path.exists() and not self.path.read_text(encoding="utf-8").lstrip().startswith(
            "{"
        ):
            self.sessions = {}
            return
        try:
            payload = load_json(self.path, default={"sessions": {}})
        except (json.JSONDecodeError, ValueError, TypeError):
            # Corrupted or half-written file: start empty rather
            # than crash the whole process.
            self.sessions = {}
            return
        for token, raw in payload.get("sessions", {}).items():
            self.sessions[token] = Session.model_validate(raw)

    def save(self) -> None:
        """Atomically persist the in-memory sessions map to disk."""
        atomic_write_json(
            self.path,
            {
                "sessions": {
                    token: session.model_dump(mode="json")
                    for token, session in self.sessions.items()
                }
            },
        )

    def create(self, user_id: str) -> Session:
        """Create a fresh session for ``user_id``."""
        now = datetime.now(UTC)
        session = Session(
            id=str(uuid4()),
            user_id=user_id,
            token=str(uuid4()),
            created_at=now,
            expires_at=now + self.timeout,
            last_seen_at=now,
        )
        with self.lock:
            self.sessions[session.token] = session
            self.save()
        return session

    def resolve(self, token: str) -> Session | None:
        """Resolve ``token`` to a live session, sliding the expiry window."""
        with self.lock:
            session = self.sessions.get(token)
            if session is None:
                return None
            now = datetime.now(UTC)
            if now > session.expires_at:
                self.sessions.pop(token, None)
                self.save()
                return None
            session.last_seen_at = now
            session.expires_at = now + self.timeout
            self.save()
            return session

    def invalidate(self, token: str) -> None:
        """Remove ``token`` from the store. No-op if unknown."""
        with self.lock:
            self.sessions.pop(token, None)
            self.save()

    def append_turn(self, token: str, turn: Turn) -> None:
        """Append ``turn`` to the session's history.

        Raises:
            AuthenticationError: If the token is invalid or expired.

        """
        with self.lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.append(turn)
            self.save()

    def load_turns(self, token: str) -> list[Turn]:
        """Return the full history for ``token``."""
        session = self.resolve(token)
        return list(session.history) if session else []

    def clear_turns(self, token: str) -> None:
        """Empty the session's conversation history.

        Raises:
            AuthenticationError: If the token is invalid or expired.

        """
        with self.lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.clear()
            self.save()


class Sessions:
    """SQLite-backed session store with sliding-window inactivity expiry.

    Mirrors :class:`JsonSessions` but persists to a SQLite table, which
    is safer for multi-process deployments. Use the :meth:`json`
    factory to get the JSON-backed equivalent.

    Attributes:
        db_path: SQLite database file path.
        timeout: Sliding inactivity window as a :class:`timedelta`.
        db: Optional shared :class:`Database`. When ``None``
            the store opens its own connections per call.

    """

    def __init__(
        self,
        db_path: str | Path,
        timeout_seconds: int = 3600,
        db: Database | None = None,
    ) -> None:
        """Initialise the store."""
        self.db_path = str(db_path)
        self.timeout = timedelta(seconds=timeout_seconds)
        self.db = db

    @classmethod
    def json(cls, path: Path, timeout_seconds: int = 3600) -> JsonSessions:
        """Construct a JSON-backed session store.

        Args:
            path: Filesystem path to the JSON file.
            timeout_seconds: Inactivity expiry window in seconds.

        Returns:
            A ready-to-use :class:`JsonSessions` instance.

        """
        return JsonSessions(path, timeout_seconds)

    async def conn(self) -> Any:
        """Return a usable connection (shared or fresh)."""
        try:
            import aiosqlite
        except ImportError:
            raise MissingDepError(
                "aiosqlite",
                "pip install raghub[auth]",
            ) from None
        if self.db is not None:
            return self.db.connection
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def maybe_commit_close(self, conn: Any) -> None:
        """Commit and close ``conn`` unless we share a manager."""
        if self.db is None:
            await conn.commit()
            await conn.close()

    async def initialize(self) -> None:
        """Create the ``sessions`` table and add the ``overrides`` column when absent."""
        conn = await self.conn()
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                history TEXT DEFAULT '[]'
            );
            """
        )
        cursor = await conn.execute("PRAGMA table_info(sessions)")
        columns = await cursor.fetchall()
        if not any(column[1] == "overrides" for column in columns):
            await conn.execute("ALTER TABLE sessions ADD COLUMN overrides TEXT DEFAULT '{}'")
        if self.db is None:
            await conn.commit()
            await conn.close()

    async def create_session_record(self, session: Session) -> None:
        """Insert a full :class:`Session` including its history."""
        conn = await self.conn()
        await conn.execute(
            """
            INSERT INTO sessions (
                session_id, user_id, token,
                created_at, expires_at, last_seen_at,
                history, overrides
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                serialize_overrides(session.overrides),
            ),
        )
        await self.maybe_commit_close(conn)

    async def create_session(self, user_id: str) -> Session:
        """Create and persist a new session."""
        now = datetime.now(UTC)
        session = Session(
            id=str(uuid4()),
            user_id=user_id,
            token=str(uuid4()),
            created_at=now,
            expires_at=now + self.timeout,
            last_seen_at=now,
        )
        conn = await self.conn()
        await conn.execute(
            """
            INSERT INTO sessions
                (session_id, user_id, token, created_at,
                 expires_at, last_seen_at, history, overrides)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([]),
                serialize_overrides(session.overrides),
            ),
        )
        await self.maybe_commit_close(conn)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Look up a session by primary key."""
        conn = await self.conn()
        cursor = await conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return None
        return self.row_to_session(row)

    async def get_by_token(self, token: str) -> Session | None:
        """Look up a session by bearer token, with sliding expiry."""
        conn = await self.conn()
        cursor = await conn.execute("SELECT * FROM sessions WHERE token = ?", (token,))
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return None
        session = self.row_to_session(row)
        now = datetime.now(UTC)
        if now > session.expires_at:
            await conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session.id,),
            )
            await self.maybe_commit_close(conn)
            return None
        session.last_seen_at = now
        session.expires_at = now + self.timeout
        await conn.execute(
            """
            UPDATE sessions
            SET last_seen_at = ?, expires_at = ?, history = ?, overrides = ?
            WHERE session_id = ?
            """,
            (
                session.last_seen_at.isoformat(),
                session.expires_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                serialize_overrides(session.overrides),
                session.id,
            ),
        )
        await self.maybe_commit_close(conn)
        return session

    async def update_session(self, session: Session) -> None:
        """Overwrite a session row with the supplied record."""
        conn = await self.conn()
        await conn.execute(
            """
            UPDATE sessions
            SET user_id = ?, token = ?, created_at = ?, expires_at = ?,
                last_seen_at = ?, history = ?, overrides = ?
            WHERE session_id = ?
            """,
            (
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                serialize_overrides(session.overrides),
                session.id,
            ),
        )
        await self.maybe_commit_close(conn)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session by primary key. No-op if unknown."""
        conn = await self.conn()
        await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self.maybe_commit_close(conn)

    async def get_overrides(self, session_id: str) -> dict[str, Any]:
        """Return the session's ``overrides`` mapping, or ``{}``."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT overrides FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return {}
        raw = row[0] if not __keyed(row) else row["overrides"]
        if not raw:
            return {}
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}

    async def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> None:
        """Replace the session's ``overrides`` mapping. No-op if unknown."""
        conn = await self.conn()
        cursor = await conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return
        await conn.execute(
            "UPDATE sessions SET overrides = ? WHERE session_id = ?",
            (serialize_overrides(overrides), session_id),
        )
        await self.maybe_commit_close(conn)

    async def append_history(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session's history. No-op if unknown."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return
        history = json.loads(row["history"])
        history.append(turn.model_dump(mode="json"))
        await conn.execute(
            "UPDATE sessions SET history = ? WHERE session_id = ?",
            (json.dumps(history), session_id),
        )
        await self.maybe_commit_close(conn)

    async def get_history(self, session_id: str) -> list[Turn]:
        """Return the full history of a session."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return []
        history = json.loads(row["history"])
        return [Turn.model_validate(t) for t in history]

    # ------------------------------------------------------------------
    # SessionStore protocol aliases
    # ------------------------------------------------------------------

    async def create(self, user_id: str) -> Session:
        """Protocol-conformant alias for :meth:`create_session`."""
        return await self.create_session(user_id)

    async def resolve(self, token: str) -> Session | None:
        """Protocol-conformant alias for :meth:`get_by_token`."""
        return await self.get_by_token(token)

    async def invalidate(self, token: str) -> None:
        """Protocol-conformant alias — deletes the session for ``token``."""
        session = await self.get_by_token(token)
        if session is not None:
            await self.delete_session(session.id)

    @staticmethod
    def row_to_session(row: Any) -> Session:
        """Hydrate a :class:`Session` from a SQLite row."""
        row_keys = row.keys() if hasattr(row, "keys") else row
        history_raw = row["history"] if "history" in row_keys else "[]"
        overrides_raw = row["overrides"] if "overrides" in row_keys else "{}"
        history = json.loads(history_raw) if history_raw else []
        overrides = json.loads(overrides_raw) if overrides_raw else {}
        return Session(
            id=row["session_id"],
            user_id=row["user_id"],
            token=row["token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
            history=[Turn.model_validate(t) for t in history],
            overrides=overrides if isinstance(overrides, dict) else {},
        )


# ---------------------------------------------------------------------------
# Migration utility
# ---------------------------------------------------------------------------


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
    import raghub.repos as repositories

    registry = repositories.DocStore(db_path)
    await registry.initialize()

    documents = Documents(Path(registry_path))
    all_versions = [doc for versions in documents.documents.values() for doc in versions]
    for doc in tqdm(
        all_versions,
        desc="Migrating documents",
        disable=not show_progress,
        unit="doc",
    ):
        await registry.save(doc)

    session_repo = repositories.SessionStore(db_path)
    await session_repo.initialize()

    json_sessions = Sessions.json(Path(sessions_path), timeout_seconds=3600)
    for session in tqdm(
        list(json_sessions.sessions.values()),
        desc="Migrating sessions",
        disable=not show_progress,
        unit="session",
    ):
        await session_repo.create_from_record(session)


__all__ = [
    "Database",
    "Documents",
    "ImageStore",
    "JsonSessions",
    "Sessions",
    "Snapshot",
    "migrate_from_json",
]
