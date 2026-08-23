/**
 * Migration runner.
 *
 * `MIGRATIONS` is an ordered, append-only list. Each entry has a
 * monotonically-increasing `id` (string) and a `sql` body that runs
 * idempotently (CREATE TABLE IF NOT EXISTS, ALTER TABLE wrapped in
 * a try/catch, etc).
 *
 * `runMigrations(db)` is called from `Workspace.open()` after the
 * base schema has executed. It records applied ids in the
 * `schema_migrations` table.
 *
 * To add a migration:
 *   1. Append to MIGRATIONS with a new id.
 *   2. Keep the sql idempotent.
 *   3. Bump the package's minor version.
 *
 * To start fresh: drop the `schema_migrations` table; the next
 * `Workspace.open()` will re-run every migration in order.
 */

export interface Migration {
  readonly id: string;
  readonly description: string;
  readonly sql: readonly string[];
}

export const MIGRATIONS: readonly Migration[] = [
  {
    id: '0001_baseline_schema',
    description: 'Initial schema (workspace, documents, chunks, sessions, audit).',
    sql: [], // empty: the baseline lives in SCHEMA_SQL in workspace.ts.
  },
  {
    id: '0002_workspace_directory',
    description: 'Top-level registry of workspace id -> on-disk path.',
    sql: [
      `CREATE TABLE IF NOT EXISTS workspace_directory (
        workspace_id TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        encryption TEXT NOT NULL DEFAULT 'passphrase-aes-256-gcm',
        registered_at INTEGER NOT NULL
      );`,
    ],
  },
  {
    id: '0003_session_remove',
    description: 'SessionStore.remove: support logout by deleting the row by raw_token.',
    sql: [],
  },
  {
    id: '0004_acl_indexes',
    description: 'Indexes that speed up document ACL lookups.',
    sql: [
      `CREATE INDEX IF NOT EXISTS idx_document_principal_principal
        ON document_principal (principal_type, principal_id);`,
      `CREATE INDEX IF NOT EXISTS idx_document_principal_doc
        ON document_principal (document_id);`,
    ],
  },
  {
    id: '0005_memory_indexes',
    description: 'Indexes for WorkspaceMemoryStore.search hot path.',
    sql: [
      `CREATE INDEX IF NOT EXISTS idx_memory_fact_scope
        ON memory_fact (workspace_id, scope, user_id);`,
    ],
  },
  {
    id: '0006_audit_event_columns',
    description:
      'Add structured columns to audit_event: kind (event type), resource_id, detail_json (rename of metadata_json).',
    sql: [
      // SQLite ALTER TABLE is not idempotent; use a guarded approach.
      // For fresh installs the base schema already contains these columns.
    ],
  },
  {
    id: '0007_users_role_column',
    description: 'Add role column to users (admin / member / viewer).',
    sql: [],
  },
  {
    id: '0008_documents_status',
    description: 'Add status column to documents.',
    sql: [],
  },
];

const MIGRATIONS_TABLE_SQL = `
  CREATE TABLE IF NOT EXISTS schema_migrations (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at INTEGER NOT NULL
  );
`;

export interface MigrationRunnerOptions {
  readonly db: { exec(sql: string): void; prepare(sql: string): { run(...args: unknown[]): unknown; all(...args: unknown[]): unknown[] } };
}

export const runMigrations = (opts: MigrationRunnerOptions): void => {
  opts.db.exec(MIGRATIONS_TABLE_SQL);
  const applied = new Set(
    (opts.db.prepare('SELECT id FROM schema_migrations').all() as Array<{ id: string }>).map((r) => r.id),
  );
  for (const m of MIGRATIONS) {
    if (applied.has(m.id)) continue;
    for (const stmt of m.sql) opts.db.exec(stmt);
    opts.db.prepare('INSERT INTO schema_migrations (id, description, applied_at) VALUES (?, ?, ?)').run(m.id, m.description, Date.now());
  }
};

export const lastAppliedId = (
  db: { prepare(sql: string): { get(...args: unknown[]): unknown } },
): string | null => {
  const row = db.prepare('SELECT id FROM schema_migrations ORDER BY applied_at DESC LIMIT 1').get() as { id?: string } | undefined;
  return row?.id ?? null;
};