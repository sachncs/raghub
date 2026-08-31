/**
 * WorkspaceRegistry — top-level registry of all workspaces on this
 * host. Stored at `~/.revex/registry.db` (or wherever
 * `REVEX_HOME` points). Maps `workspace_id → absolute .db path`.
 *
 * One process owns one registry. `Workspace.open()` is unaware of
 * the registry — it's a single SQLite file. The API layer keeps a
 * `WorkspaceRegistry` open alongside every active workspace handle
 * so it can resolve a JWT's `workspace_id` to a `.db` path.
 *
 * Single-tenant deployments (the default) can ignore this and
 * just call `Workspace.open({ path })` directly.
 */

import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

import type { WorkspaceId } from './domain/index.js';

export interface WorkspaceRegistryEntry {
  readonly workspaceId: WorkspaceId;
  readonly path: string;
  readonly encryption: 'plaintext' | 'passphrase-aes-256-gcm';
  readonly registeredAt: Date;
}

export interface WorkspaceRegistry {
  register(input: { workspaceId: WorkspaceId; path: string; encryption: 'plaintext' | 'passphrase-aes-256-gcm' }): Promise<void>;
  resolve(workspaceId: WorkspaceId): Promise<WorkspaceRegistryEntry | null>;
  unregister(workspaceId: WorkspaceId): Promise<void>;
  list(): Promise<readonly WorkspaceRegistryEntry[]>;
  close(): Promise<void>;
}

export interface FileWorkspaceRegistryOptions {
  readonly registryPath: string;
}

interface DatabaseLike {
  prepare(sql: string): {
    run(...args: unknown[]): unknown;
    get(...args: unknown[]): unknown;
    all(...args: unknown[]): unknown[];
  };
  exec(sql: string): void;
  pragma?(k: string): unknown;
  close(): void;
}

interface SqliteFactory {
  (path: string): DatabaseLike;
}

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS workspace_directory (
    workspace_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    encryption TEXT NOT NULL DEFAULT 'passphrase-aes-256-gcm',
    registered_at INTEGER NOT NULL
  );
`;

const rowToEntry = (row: Record<string, unknown>): WorkspaceRegistryEntry => ({
  workspaceId: String(row['workspace_id']) as WorkspaceId,
  path: String(row['path']),
  encryption: (row['encryption'] as 'plaintext' | 'passphrase-aes-256-gcm') ?? 'passphrase-aes-256-gcm',
  registeredAt: new Date(Number(row['registered_at'])),
});

export const defaultRegistryPath = (homeDir: string): string => join(homeDir, 'registry.db');

export const openFileWorkspaceRegistry = async (
  opts: FileWorkspaceRegistryOptions,
  sqlite: SqliteFactory,
): Promise<WorkspaceRegistry> => {
  const dir = dirname(opts.registryPath);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  const db = sqlite(opts.registryPath);
  if (db.pragma) db.pragma('journal_mode = WAL');
  if (db.pragma) db.pragma('busy_timeout = 5000');
  db.exec(SCHEMA);

  return {
    async register(input): Promise<void> {
      db.prepare(
        `INSERT INTO workspace_directory (workspace_id, path, encryption, registered_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(workspace_id) DO UPDATE SET path = excluded.path, encryption = excluded.encryption`,
      ).run(input.workspaceId, input.path, input.encryption, Date.now());
    },
    async resolve(workspaceId): Promise<WorkspaceRegistryEntry | null> {
      const row = db.prepare('SELECT * FROM workspace_directory WHERE workspace_id = ?').get(workspaceId) as Record<string, unknown> | undefined;
      return row ? rowToEntry(row) : null;
    },
    async unregister(workspaceId): Promise<void> {
      db.prepare('DELETE FROM workspace_directory WHERE workspace_id = ?').run(workspaceId);
    },
    async list(): Promise<readonly WorkspaceRegistryEntry[]> {
      const rows = db.prepare('SELECT * FROM workspace_directory ORDER BY registered_at ASC').all() as Record<string, unknown>[];
      return rows.map(rowToEntry);
    },
    async close(): Promise<void> {
      db.close();
    },
  };
};