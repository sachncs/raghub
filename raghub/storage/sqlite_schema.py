"""SQLite schema DDL used by the unit-of-work and store initialisers.

This module exposes a single :data:`SQLITE_SCHEMA` constant containing
the ``CREATE TABLE IF NOT EXISTS`` statements for the application's
three primary tables (``documents``, ``chunks``, ``sessions``,
``users``). It is consumed by :class:`UnitOfWork` to bootstrap an empty
database on first run.

The schema is intentionally minimal: there are no indexes beyond the
primary keys declared inline. Production deployments with high read
volume should add indexes on the hot columns (e.g. ``chunks.document_id``,
``documents.organization``) in a follow-up migration.
"""

from __future__ import annotations

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