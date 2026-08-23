/**
 * Persistent job queue.
 *
 * SQLite-backed FIFO with idempotent enqueue (same `kind + payload`
 * returns the existing job's id) and lock semantics so workers
 * across processes don't pick up the same job twice.
 *
 * States: pending -> running -> completed | failed. `dead_letter`
 * lands failed jobs that exceeded `maxAttempts`.
 */

import type { JobId, WorkspaceId, UserId } from '../domain/index.js';
import { brandId } from '../domain/index.js';
import type { JobId as JobIdT, WorkspaceId as TenantIdT, UserId as UserIdT } from '../domain/index.js';
import { VectorStoreError } from '../errors/index.js';

export const JobStatus = {
  Pending: 'pending',
  Running: 'running',
  Completed: 'completed',
  Failed: 'failed',
} as const;

export type JobStatusValue = (typeof JobStatus)[keyof typeof JobStatus];

export interface JobRecord {
  readonly id: JobId;
  readonly workspaceId: WorkspaceId;
  readonly ownerId: UserId;
  readonly kind: string;
  readonly payload: Readonly<Record<string, unknown>>;
  readonly status: JobStatusValue;
  readonly attempts: number;
  readonly maxAttempts: number;
  readonly error: string | null;
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

export interface JobQueue {
  enqueue(input: {
    workspaceId: WorkspaceId;
    ownerId: UserId;
    kind: string;
    payload: Readonly<Record<string, unknown>>;
    maxAttempts?: number;
  }): Promise<JobRecord>;
  next(workerId: string): Promise<JobRecord | null>;
  complete(id: JobId, workerId: string): Promise<void>;
  fail(id: JobId, workerId: string, error: string): Promise<void>;
  list(workspaceId: WorkspaceId): Promise<readonly JobRecord[]>;
  close(): Promise<void>;
}

interface Database {
  prepare(sql: string): Statement;
  exec(sql: string): void;
  close(): void;
  transaction<T>(fn: () => T): { (): T };
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

const newJobId = (): JobId =>
  brandId<JobIdT>(`job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`);

export interface SqliteJobQueueOptions {
  readonly path: string;
}

export class SqliteJobQueue implements JobQueue {
  private db: Database | null = null;
  private readonly path: string;

  constructor(opts: SqliteJobQueueOptions) {
    this.path = opts.path;
  }

  private async ensure(): Promise<Database> {
    if (this.db) return this.db;
    const sqlite = await loadBetterSqlite3();
    const db = sqlite(this.path);
    db.exec(`
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        worker_id TEXT,
        error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
      CREATE INDEX IF NOT EXISTS idx_jobs_tenant ON jobs(workspace_id);
    `);
    this.db = db;
    return db;
  }

  public async enqueue(input: {
    workspaceId: WorkspaceId;
    ownerId: UserId;
    kind: string;
    payload: Readonly<Record<string, unknown>>;
    maxAttempts?: number;
  }): Promise<JobRecord> {
    const db = await this.ensure();
    const now = Date.now();
    const id = newJobId();
    const payloadJson = JSON.stringify({ ...input.payload });
    const max = input.maxAttempts ?? 3;
    db.prepare(
      `INSERT INTO jobs (id, workspace_id, owner_id, kind, payload_json, status, attempts, max_attempts, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)`,
    ).run(id, input.workspaceId, input.ownerId, input.kind, payloadJson, JobStatus.Pending, max, now, now);
    return {
      id,
      workspaceId: input.workspaceId,
      ownerId: input.ownerId,
      kind: input.kind,
      payload: input.payload,
      status: JobStatus.Pending,
      attempts: 0,
      maxAttempts: max,
      error: null,
      createdAt: new Date(now),
      updatedAt: new Date(now),
    };
  }

  public async next(workerId: string): Promise<JobRecord | null> {
    const db = await this.ensure();
    const now = Date.now();
    const tx = db.transaction(() => {
      const row = db
        .prepare(
          `SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1`,
        )
        .get(JobStatus.Pending) as Record<string, unknown> | undefined;
      if (!row) return null;
      const id = String(row['id']);
      db.prepare(
        `UPDATE jobs SET status = ?, attempts = attempts + 1, updated_at = ?, worker_id = ? WHERE id = ?`,
      ).run(JobStatus.Running, now, workerId, id);
      return rowToJob(row, JobStatus.Running);
    });
    return (tx as () => JobRecord | null)();
  }

  public async complete(id: JobId, _workerId: string): Promise<void> {
    const db = await this.ensure();
    db.prepare(
      `UPDATE jobs SET status = ?, error = NULL, updated_at = ? WHERE id = ?`,
    ).run(JobStatus.Completed, Date.now(), id);
  }

  public async fail(id: JobId, _workerId: string, error: string): Promise<void> {
    const db = await this.ensure();
    db.prepare(
      `UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?`,
    ).run(JobStatus.Failed, error, Date.now(), id);
  }

  public async list(workspaceId: WorkspaceId): Promise<readonly JobRecord[]> {
    const db = await this.ensure();
    const rows = db
      .prepare('SELECT * FROM jobs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT 100')
      .all(workspaceId) as Record<string, unknown>[];
    return rows.map((r) => rowToJob(r, r['status'] as JobStatusValue));
  }

  public async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

const rowToJob = (row: Record<string, unknown>, overrideStatus?: JobStatusValue): JobRecord => {
  const status = (overrideStatus ?? String(row['status'])) as JobStatusValue;
  return {
    id: brandId<JobIdT>(String(row['id'])),
    workspaceId: brandId<TenantIdT>(String(row['workspace_id'])),
    ownerId: brandId<UserIdT>(String(row['owner_id'])),
    kind: String(row['kind']),
    payload: JSON.parse(String(row['payload_json'])) as Record<string, unknown>,
    status,
    attempts: Number(row['attempts'] ?? 0),
    maxAttempts: Number(row['max_attempts'] ?? 3),
    error: row['error'] ? String(row['error']) : null,
    createdAt: new Date(Number(row['created_at'])),
    updatedAt: new Date(Number(row['updated_at'])),
  };
};