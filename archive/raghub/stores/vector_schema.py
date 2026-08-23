"""Canonical database schema for raghub's primary stores.

Single source of truth (R8) for the documents / sessions / chunks /
queue / feedback / audit tables. Imported by every store
implementation; ad-hoc SQL strings scattered across modules are
forbidden.
"""

from __future__ import annotations

__all__ = [
    "AUDIT_SCHEMA_SQL",
    "DOCUMENTS_SCHEMA_SQL",
    "FEEDBACK_SCHEMA_SQL",
    "SESSIONS_SCHEMA_SQL",
]


DOCUMENTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    filename TEXT NOT NULL,
    checksum TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    status TEXT NOT NULL,
    user_id TEXT NOT NULL,
    organization TEXT,
    department TEXT,
    classification TEXT NOT NULL DEFAULT 'INTERNAL',
    visibility TEXT NOT NULL DEFAULT 'PRIVATE',
    owner TEXT,
    source_uri TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum);
"""


SESSIONS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    history TEXT DEFAULT '[]',
    overrides TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
"""


FEEDBACK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raghub_feedback (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    chunk_id TEXT,
    answer_id TEXT,
    user_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE (session_id, query_id, chunk_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_tenant_chunk
    ON raghub_feedback (tenant_id, chunk_id);
"""


AUDIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raghub_audit (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    route TEXT NOT NULL,
    decision TEXT NOT NULL,
    retry_after_seconds INTEGER,
    timestamp TEXT NOT NULL,
    signature TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_id
    ON raghub_audit (tenant_id, timestamp);
"""
