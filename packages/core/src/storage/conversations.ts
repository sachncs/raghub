/**
 * Conversation store — sqlite-backed, shared Database handle.
 *
 * Sliding-window history: `history(sessionId, maxTurns)` returns the
 * last `maxTurns` rows in chronological order.
 */

import type { SessionId, WorkspaceId, Turn, UserId } from '../domain/index.js';
import { Turn as TurnClass, TurnRole } from '../domain/index.js';
import { brandId } from '../domain/index.js';
import type { Database } from '../workspace.js';

export interface TurnInput {
  readonly role: keyof typeof TurnRole;
  readonly content: string;
}

export interface ConversationStore {
  append(
    sessionId: SessionId,
    workspaceId: WorkspaceId,
    userId: UserId,
    turn: TurnInput,
  ): Promise<Turn>;
  history(sessionId: SessionId, workspaceId: WorkspaceId, maxTurns: number): Promise<readonly Turn[]>;
  clear(sessionId: SessionId, workspaceId: WorkspaceId): Promise<void>;
  close(): Promise<void>;
}

export interface SqliteConversationStoreOptions {
  readonly db: Database;
}

export class SqliteConversationStore implements ConversationStore {
  private readonly db: Database;

  constructor(opts: SqliteConversationStoreOptions) {
    this.db = opts.db;
  }

  public async append(
    sessionId: SessionId,
    workspaceId: WorkspaceId,
    userId: UserId,
    turn: TurnInput,
  ): Promise<Turn> {
    const now = Date.now();
    const role = TurnRole[turn.role];
    this.db
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
    const rows = this.db
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
    this.db
      .prepare('DELETE FROM turns WHERE session_id = ? AND workspace_id = ?')
      .run(sessionId, workspaceId);
  }

  public async close(): Promise<void> {
    // No-op.
  }
}
