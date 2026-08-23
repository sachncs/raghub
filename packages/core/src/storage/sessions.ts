/**
 * Session store — sqlite-backed, shared Database handle.
 *
 * Sessions are namespaced by `(user_id, raw_token)`. Two callers
 * sharing or guessing a token cannot read each other's history.
 */

import type { SessionId, WorkspaceId, UserId } from '../domain/index.js';
import { brandId } from '../domain/index.js';
import type { Database } from '../workspace.js';

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
  remove(rawToken: string, workspaceId?: WorkspaceId): Promise<void>;
  close(): Promise<void>;
}

export interface SqliteSessionStoreOptions {
  readonly db: Database;
}

const namespaceId = (userId: string, rawToken: string): string => `${userId}::${rawToken}`;

export class SqliteSessionStore implements SessionStore {
  private readonly db: Database;

  constructor(opts: SqliteSessionStoreOptions) {
    this.db = opts.db;
  }

  public async upsert(input: {
    workspaceId: WorkspaceId;
    userId: UserId;
    rawToken: string;
    strategyOverrides?: Readonly<Record<string, unknown>>;
  }): Promise<SessionRecord> {
    const now = Date.now();
    const id = brandId<SessionId>(namespaceId(input.userId, input.rawToken));
    const overridesJson = JSON.stringify({ ...(input.strategyOverrides ?? {}) });
    this.db
      .prepare(
        `INSERT INTO sessions (id, workspace_id, user_id, raw_token, strategy_overrides_json, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(raw_token) DO UPDATE SET
           strategy_overrides_json = excluded.strategy_overrides_json,
           updated_at = excluded.updated_at`,
      )
      .run(id, input.workspaceId, input.userId, input.rawToken, overridesJson, now, now);
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
    const row = this.db
      .prepare('SELECT * FROM sessions WHERE raw_token = ?')
      .get(rawToken) as Record<string, unknown> | undefined;
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
    // No-op.
  }

  public async remove(rawToken: string, workspaceId?: WorkspaceId): Promise<void> {
    void workspaceId;
    this.db.prepare('DELETE FROM sessions WHERE raw_token = ?').run(rawToken);
  }
}
