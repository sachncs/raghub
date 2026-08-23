/**
 * Session store.
 *
 * Stores session metadata + per-session overrides (the strategy
 * overrides PATCHed via /v1/me/strategy land here, scoped to the
 * session id). Sessions are namespaced by (userId, rawToken) so two
 * callers who guess or share a token cannot read each other's
 * history.
 */

import type { SessionId, WorkspaceId, UserId } from '../domain/index.js';
import { brandId } from '../domain/index.js';
import { VectorStoreError } from '../errors/index.js';

export interface SessionRecord {
  readonly id: SessionId;
  readonly workspaceId: WorkspaceId;
  readonly userId: UserId;
  readonly strategyOverrides: Readonly<Record<string, unknown>>;
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

export interface SessionStore {
  upsert(input: {
    workspaceId: WorkspaceId;
    userId: UserId;
    rawToken: string;
    strategyOverrides?: Readonly<Record<string, unknown>>;
  }): Promise<SessionRecord>;
  get(rawToken: string): Promise<SessionRecord | null>;
  close(): Promise<void>;
}

interface Database {
  prepare(sql: string): Statement;
  exec(sql: string): void;
  close(): void;
}

interface Statement {
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
  run(...params: unknown[]): { changes: number; lastInsertRowid: number | bigint };
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadBetterSqlite3 = async (): Promise<(filename: string) => Database> => {
  try {
    const mod = (await dynamicImport('better-sqlite3')) as {
      default: (filename: string) => Database;
    };
    return mod.default;
  } catch (cause) {
    throw new VectorStoreError('better-sqlite3 is not installed', {
      cause,
      details: { hint: 'pnpm add better-sqlite3' },
    });
  }
};

const namespaceId = (userId: string, rawToken: string): string => `${userId}::${rawToken}`;

export interface SqliteSessionStoreOptions {
  readonly path: string;
}

export class SqliteSessionStore implements SessionStore {
  private db: Database | null = null;
  private readonly path: string;

  constructor(opts: SqliteSessionStoreOptions) {
    this.path = opts.path;
  }

  private async ensure(): Promise<Database> {
    if (this.db) return this.db;
    const sqlite = await loadBetterSqlite3();
    const db = sqlite(this.path);
    db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        raw_token TEXT NOT NULL UNIQUE,
        strategy_overrides_json TEXT NOT NULL DEFAULT '{}',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
    `);
    this.db = db;
    return db;
  }

  public async upsert(input: {
    workspaceId: WorkspaceId;
    userId: UserId;
    rawToken: string;
    strategyOverrides?: Readonly<Record<string, unknown>>;
  }): Promise<SessionRecord> {
    const db = await this.ensure();
    const now = Date.now();
    const id = brandId<SessionId>(namespaceId(input.userId, input.rawToken));
    const overridesJson = JSON.stringify({ ...(input.strategyOverrides ?? {}) });
    db.prepare(
      `INSERT INTO sessions (id, workspace_id, user_id, raw_token, strategy_overrides_json, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(raw_token) DO UPDATE SET
         strategy_overrides_json = excluded.strategy_overrides_json,
         updated_at = excluded.updated_at`,
    ).run(id, input.workspaceId, input.userId, input.rawToken, overridesJson, now, now);
    return {
      id,
      workspaceId: input.workspaceId,
      userId: input.userId,
      strategyOverrides: input.strategyOverrides ?? {},
      createdAt: new Date(now),
      updatedAt: new Date(now),
    };
  }

  public async get(rawToken: string): Promise<SessionRecord | null> {
    const db = await this.ensure();
    const row = db.prepare('SELECT * FROM sessions WHERE raw_token = ?').get(rawToken) as
      | Record<string, unknown>
      | undefined;
    if (!row) return null;
    return {
      id: brandId<SessionId>(String(row['id'])),
      workspaceId: brandId<WorkspaceId>(String(row['workspace_id'])),
      userId: brandId<UserId>(String(row['user_id'])),
      strategyOverrides: JSON.parse(String(row['strategy_overrides_json'] ?? '{}')) as Record<string, unknown>,
      createdAt: new Date(Number(row['created_at'])),
      updatedAt: new Date(Number(row['updated_at'])),
    };
  }

  public async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}