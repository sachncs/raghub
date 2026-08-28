/**
 * Workspace — the single SQLite handle every store shares.
 *
 * `Workspace.open(path)` opens (or creates) the database, runs every
 * idempotent `CREATE TABLE IF NOT EXISTS` migration, and returns a
 * `Workspace` whose `.db` handle is the one source of truth.
 *
 * C-03 only handles the unencrypted path. C-04 adds passphrase +
 * AES-256-GCM at-rest encryption; the public surface here is
 * designed so encryption just plugs in.
 *
 * Every `Sqlite*Store` takes the shared `Database` handle via the
 * `fromWorkspace(workspace)` factory. Stores that share one
 * workspace share one SQLite connection — no more `swapDbPath`,
 * no more separate `*.db` files for users, documents, sessions,
 * conversations, jobs.
 */

import { randomUUID } from 'node:crypto';

import { runMigrations } from './migrations.js';

export interface Database {
  prepare(sql: string): Statement;
  exec(sql: string): void;
  close(): void;
  pragma?(source: string): unknown;
  /** Wraps fn in a transaction. Optional because not every driver supports it. */
  transaction?<T>(fn: () => T): () => T;
}

export interface Statement {
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
  run(...params: unknown[]): { changes: number; lastInsertRowid: number | bigint };
}

export interface WorkspaceOptions {
  /**
   * Path to the SQLite file. `:memory:` for tests. If the file
   * does not exist, it is created and the schema is initialised.
   */
  readonly path: string;
}

const SCHEMA_SQL = `
  CREATE TABLE IF NOT EXISTS workspace_settings (
    key TEXT PRIMARY KEY,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    updated_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS workspace_keycheck (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    salt BLOB NOT NULL,
    kdf_params TEXT NOT NULL,
    verifier_nonce BLOB NOT NULL,
    verifier_ciphertext BLOB NOT NULL,
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS workspace (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT UNIQUE NOT NULL,
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS workspace_member (
    user_id TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('owner','admin','member','viewer')),
    joined_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS role (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS role_member (
    role_id TEXT NOT NULL,
    principal_type TEXT NOT NULL CHECK (principal_type IN ('user','group')),
    principal_id TEXT NOT NULL,
    PRIMARY KEY (role_id, principal_type, principal_id)
  );

  CREATE TABLE IF NOT EXISTS workspace_group (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS workspace_group_member (
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    PRIMARY KEY (group_id, user_id)
  );

  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    is_admin INTEGER NOT NULL DEFAULT 0,
    allowed_companies_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    owner_id TEXT NOT NULL,
    group_id TEXT,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    hash TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (workspace_id, hash)
  );

  CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    owner_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'text',
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
  );

  CREATE INDEX IF NOT EXISTS idx_chunks_workspace ON chunks (workspace_id);
  CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
  CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks (workspace_id, owner_id);

  CREATE TABLE IF NOT EXISTS document_principal (
    document_id TEXT NOT NULL,
    principal_type TEXT NOT NULL CHECK (principal_type IN ('user','role','group')),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK (permission IN ('read','admin')),
    granted_by TEXT NOT NULL,
    granted_at INTEGER NOT NULL,
    PRIMARY KEY (document_id, principal_type, principal_id, permission)
  );
  CREATE INDEX IF NOT EXISTS idx_document_principal_principal
    ON document_principal (principal_type, principal_id);

  CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    user_id TEXT NOT NULL,
    raw_token TEXT NOT NULL UNIQUE,
    strategy_overrides_json TEXT NOT NULL DEFAULT '{}',
    expires_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, workspace_id, id DESC);

  CREATE TABLE IF NOT EXISTS memory_fact (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    scope TEXT NOT NULL CHECK (scope IN ('user','workspace')),
    user_id TEXT,
    content TEXT NOT NULL,
    embedding BLOB,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_memory_fact_workspace ON memory_fact (workspace_id, scope);

  CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    owner_id TEXT NOT NULL,
    document_id TEXT,
    status TEXT NOT NULL,
    error TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'document.ingest',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3
  );
  CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status
    ON ingestion_jobs (status, created_at);

  CREATE TABLE IF NOT EXISTS graph_entities (
    name TEXT NOT NULL,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (name, workspace_id, chunk_id)
  );
  CREATE INDEX IF NOT EXISTS idx_graph_entities_workspace ON graph_entities (workspace_id);

  CREATE TABLE IF NOT EXISTS graph_edges (
    workspace_id INTEGER NOT NULL DEFAULT 1,
    from_name TEXT NOT NULL,
    to_name TEXT NOT NULL,
    weight INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (workspace_id, from_name, to_name)
  );

  CREATE TABLE IF NOT EXISTS trace_corpus (
    trace_id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    user_id TEXT,
    source_problem TEXT NOT NULL,
    raw TEXT NOT NULL,
    struct TEXT,
    semantic TEXT,
    reflect TEXT,
    embedding BLOB,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_trace_corpus_workspace ON trace_corpus (workspace_id);

  CREATE TABLE IF NOT EXISTS audit_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL DEFAULT 1,
    actor_id TEXT,
    kind TEXT NOT NULL,
    resource_id TEXT,
    detail_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_audit_workspace ON audit_event (workspace_id, created_at DESC);
`;

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadBetterSqlite3 = async (): Promise<(filename: string) => Database> => {
  const mod = (await dynamicImport('better-sqlite3')) as {
    default: (filename: string) => Database;
  };
  return mod.default;
};

export interface WorkspaceHandle {
  readonly path: string;
  readonly db: Database;
  readonly id: string;
  close(): void;
}

export type Workspace = WorkspaceHandle;

export const openWorkspace = async (opts: WorkspaceOptions): Promise<WorkspaceHandle> => {
  const sqlite = await loadBetterSqlite3();
  const db = sqlite(opts.path);
  if (db.pragma) db.pragma('journal_mode = WAL');
  if (db.pragma) db.pragma('synchronous = NORMAL');
  if (db.pragma) db.pragma('busy_timeout = 5000');
  if (db.pragma) db.pragma('foreign_keys = ON');
  if (db.pragma) db.pragma('temp_store = MEMORY');
  if (db.pragma) db.pragma('cache_size = -64000');
  db.exec(SCHEMA_SQL);
  runMigrations({ db: { exec: (s) => db.exec(s), prepare: (s) => db.prepare(s) } });

  const id = String((db.prepare('SELECT id FROM workspace WHERE id = 1').get() as { id?: number } | undefined)?.id ?? '');
  if (id === '') {
    db.prepare('INSERT OR IGNORE INTO workspace (id, name, created_at) VALUES (1, ?, ?)').run(
      'workspace',
      Date.now(),
    );
  }

  return {
    path: opts.path,
    db,
    id: randomUUID(),
    close: () => db.close(),
  } satisfies WorkspaceHandle;
};
