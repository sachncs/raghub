/**
 * Conversation store.
 *
 * SQLite-backed turn history. Each turn is appended on the active
 * session; the store trims by sliding window when the session
 * exceeds `maxTurns`.
 */

import type { SessionId, WorkspaceId, Turn, UserId } from '../domain/index.js';
import { brandId, Turn as TurnClass, TurnRole } from '../domain/index.js';
import { VectorStoreError } from '../errors/index.js';

export interface TurnInput {
  readonly role: keyof typeof TurnRole;
  readonly content: string;
}

export interface ConversationStore {
  append(sessionId: SessionId, workspaceId: WorkspaceId, userId: UserId, turn: TurnInput): Promise<Turn>;
  history(sessionId: SessionId, workspaceId: WorkspaceId, maxTurns: number): Promise<readonly Turn[]>;
  clear(sessionId: SessionId, workspaceId: WorkspaceId): Promise<void>;
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

export interface SqliteConversationStoreOptions {
  readonly path: string;
}

export class SqliteConversationStore implements ConversationStore {
  private db: Database | null = null;
  private readonly path: string;

  constructor(opts: SqliteConversationStoreOptions) {
    this.path = opts.path;
  }

  private async ensure(): Promise<Database> {
    if (this.db) return this.db;
    const sqlite = await loadBetterSqlite3();
    const db = sqlite(this.path);
    db.exec(`
      CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, workspace_id, id DESC);
    `);
    this.db = db;
    return db;
  }

  public async append(
    sessionId: SessionId,
    workspaceId: WorkspaceId,
    userId: UserId,
    turn: TurnInput,
  ): Promise<Turn> {
    const db = await this.ensure();
    const now = Date.now();
    const role = TurnRole[turn.role];
    const r = db
      .prepare(
        `INSERT INTO turns (session_id, workspace_id, user_id, role, content, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(sessionId, workspaceId, userId, role, turn.content, now);
    return new TurnClass({
      sessionId,
      workspaceId,
      userId,
      role,
      content: turn.content,
      createdAt: new Date(now),
    });
  }

  public async history(
    sessionId: SessionId,
    workspaceId: WorkspaceId,
    maxTurns: number,
  ): Promise<readonly Turn[]> {
    const db = await this.ensure();
    const rows = db
      .prepare(
        `SELECT * FROM turns WHERE session_id = ? AND workspace_id = ? ORDER BY id DESC LIMIT ?`,
      )
      .all(sessionId, workspaceId, maxTurns) as Record<string, unknown>[];
    return rows.map((r) => {
      const role = String(r['role']) as keyof typeof TurnRole;
      return new TurnClass({
        sessionId: brandId<SessionId>(String(r['session_id'])),
        workspaceId: brandId<WorkspaceId>(String(r['workspace_id'])),
        userId: brandId<UserId>(String(r['user_id'])),
        role: TurnRole[role] ?? TurnRole.User,
        content: String(r['content']),
        createdAt: new Date(Number(r['created_at'])),
      });
    });
  }

  public async clear(sessionId: SessionId, workspaceId: WorkspaceId): Promise<void> {
    const db = await this.ensure();
    db.prepare('DELETE FROM turns WHERE session_id = ? AND workspace_id = ?').run(sessionId, workspaceId);
  }

  public async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}