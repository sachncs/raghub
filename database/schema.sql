-- RAG SQLite schema.
--
-- Three tables cover the persisted state of the demo application:
--
-- * ``documents`` — one row per uploaded file. ``company`` is the
--   tenant tag and drives the RBAC filter.
-- * ``chunks`` — one row per text chunk extracted from a document.
--   ``page`` records the originating page/section index.
-- * ``conversations`` — append-only log of ``(user, session)`` turns
--   with ``role`` (``user``/``assistant``) and ``timestamp``.

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    title TEXT NOT NULL,
    path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    company TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    session TEXT NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

