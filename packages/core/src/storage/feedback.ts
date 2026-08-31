/**
 * SQLite-backed feedback store.
 *
 * Persists `Feedback` rows per workspace. Read APIs return rows in
 * insertion order (newest first); aggregate helpers compute
 * up/down counts and average per-question ratings.
 */

import type { Database } from '../workspace.js';

import {
  FeedbackRating,
  type Feedback,
  type FeedbackId,
  type FeedbackRatingValue,
  type TurnId,
} from '../feedback/index.js';
import type { UserId, WorkspaceId } from '../domain/index.js';
import { brandId } from '../domain/ids.js';

export interface FeedbackStore {
  add(workspaceId: WorkspaceId, ownerId: UserId, turnId: TurnId, rating: FeedbackRatingValue, comment: string | null): Feedback;
  list(workspaceId: WorkspaceId, opts?: { limit?: number; turnId?: TurnId }): readonly Feedback[];
  get(workspaceId: WorkspaceId, id: FeedbackId): Feedback | null;
  delete(workspaceId: WorkspaceId, id: FeedbackId): boolean;
  aggregate(workspaceId: WorkspaceId): FeedbackAggregate;
}

export interface FeedbackAggregate {
  readonly total: number;
  readonly up: number;
  readonly down: number;
  readonly neutral: number;
}

interface Stmt {
  run(...args: unknown[]): unknown;
  get(...args: unknown[]): unknown;
  all(...args: unknown[]): unknown[];
}

interface Row {
  readonly id: string;
  readonly workspace_id: string;
  readonly owner_id: string;
  readonly turn_id: string;
  readonly rating: string;
  readonly comment: string | null;
  readonly created_at: number;
}

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  rating TEXT NOT NULL CHECK (rating IN ('up','down','neutral')),
  comment TEXT,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_ws_created ON feedback (workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS feedback_ws_turn ON feedback (workspace_id, turn_id);
`;

export class SqliteFeedbackStore implements FeedbackStore {
  private readonly db: Database;
  private readonly addStmt: Stmt;
  private readonly listStmt: Stmt;
  private readonly listTurnStmt: Stmt;
  private readonly getStmt: Stmt;
  private readonly deleteStmt: Stmt;
  private readonly aggregateStmt: Stmt;

  constructor(opts: { readonly db: Database }) {
    this.db = opts.db;
    opts.db.exec(SCHEMA_SQL);
    const db = this.db as unknown as {
      prepare: (sql: string) => Stmt;
      exec: (sql: string) => void;
    };
    this.addStmt = db.prepare(
      `INSERT INTO feedback (id, workspace_id, owner_id, turn_id, rating, comment, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    );
    this.listStmt = db.prepare(
      `SELECT * FROM feedback WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?`,
    );
    this.listTurnStmt = db.prepare(
      `SELECT * FROM feedback WHERE workspace_id = ? AND turn_id = ? ORDER BY created_at DESC`,
    );
    this.getStmt = db.prepare(`SELECT * FROM feedback WHERE workspace_id = ? AND id = ?`);
    this.deleteStmt = db.prepare(`DELETE FROM feedback WHERE workspace_id = ? AND id = ?`);
    this.aggregateStmt = db.prepare(
      `SELECT rating, COUNT(*) AS n FROM feedback WHERE workspace_id = ? GROUP BY rating`,
    );
  }

  add(
    workspaceId: WorkspaceId,
    ownerId: UserId,
    turnId: TurnId,
    rating: FeedbackRatingValue,
    comment: string | null,
  ): Feedback {
    const id = brandId<FeedbackId>(`fb_${newId()}`);
    const createdAt = Date.now();
    this.addStmt.run(id, workspaceId, ownerId, turnId, rating, comment, createdAt);
    return {
      id,
      workspaceId,
      ownerId,
      turnId,
      rating,
      comment,
      createdAt,
    };
  }

  list(workspaceId: WorkspaceId, opts?: { limit?: number; turnId?: TurnId }): readonly Feedback[] {
    const limit = opts?.limit ?? 100;
    const rows = opts?.turnId
      ? (this.listTurnStmt.all(workspaceId, opts.turnId) as Row[])
      : (this.listStmt.all(workspaceId, limit) as Row[]);
    return rows.map(rowToFeedback);
  }

  get(workspaceId: WorkspaceId, id: FeedbackId): Feedback | null {
    const row = this.getStmt.get(workspaceId, id) as Row | undefined;
    return row ? rowToFeedback(row) : null;
  }

  delete(workspaceId: WorkspaceId, id: FeedbackId): boolean {
    const info = this.deleteStmt.run(workspaceId, id) as { changes: number };
    return info.changes > 0;
  }

  aggregate(workspaceId: WorkspaceId): FeedbackAggregate {
    const rows = this.aggregateStmt.all(workspaceId) as { rating: string; n: number }[];
    let total = 0;
    let up = 0;
    let down = 0;
    let neutral = 0;
    for (const r of rows) {
      const n = Number(r.n);
      total += n;
      if (r.rating === FeedbackRating.Up) up += n;
      else if (r.rating === FeedbackRating.Down) down += n;
      else if (r.rating === FeedbackRating.Neutral) neutral += n;
    }
    return { total, up, down, neutral };
  }
}

function rowToFeedback(r: Row): Feedback {
  return {
    id: brandId<FeedbackId>(r.id),
    workspaceId: r.workspace_id as WorkspaceId,
    ownerId: r.owner_id as UserId,
    turnId: r.turn_id as TurnId,
    rating: r.rating as FeedbackRatingValue,
    comment: r.comment,
    createdAt: r.created_at,
  };
}

function newId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID().replace(/-/g, '').slice(0, 16);
  }
  return Math.random().toString(36).slice(2, 18);
}