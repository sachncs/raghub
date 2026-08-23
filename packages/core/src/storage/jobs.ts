/**
 * Job queue — sqlite-backed, shared Database handle.
 */

import type { JobId, WorkspaceId, UserId } from '../domain/index.js';
import { brandId } from '../domain/index.js';
import { VectorStoreError } from '../errors/index.js';
import type { Database } from '../workspace.js';

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

export interface SqliteJobQueueOptions {
  readonly db: Database;
}

export class SqliteJobQueue implements JobQueue {
  private readonly db: Database;

  constructor(opts: SqliteJobQueueOptions) {
    this.db = opts.db;
  }

  public async enqueue(input: {
    workspaceId: WorkspaceId;
    ownerId: UserId;
    kind: string;
    payload: Readonly<Record<string, unknown>>;
    maxAttempts?: number;
  }): Promise<JobRecord> {
    const now = Date.now();
    const id = brandId<JobId>(`job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`);
    const payloadJson = JSON.stringify({ ...input.payload });
    const max = input.maxAttempts ?? 3;
    this.db
      .prepare(
        `INSERT INTO jobs (id, workspace_id, owner_id, kind, payload_json, status, attempts, max_attempts, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)`,
      )
      .run(id, input.workspaceId, input.ownerId, input.kind, payloadJson, JobStatus.Pending, max, now, now);
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
    const now = Date.now();
    const tx = this.db.transaction
      ? this.db.transaction(() => this.claim(workerId, now))
      : () => this.claim(workerId, now);
    return (tx as () => JobRecord | null)();
  }

  private claim(workerId: string, now: number): JobRecord | null {
    const row = this.db
      .prepare(`SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1`)
      .get(JobStatus.Pending) as Record<string, unknown> | undefined;
    if (!row) return null;
    const id = String(row['id']);
    this.db
      .prepare(
        `UPDATE jobs SET status = ?, attempts = attempts + 1, updated_at = ?, owner_id = COALESCE(owner_id, ?) WHERE id = ?`,
      )
      .run(JobStatus.Running, now, workerId, id);
    return rowToJob(row, JobStatus.Running);
  }

  public async complete(id: JobId, _workerId: string): Promise<void> {
    this.db
      .prepare('UPDATE jobs SET status = ?, error = NULL, updated_at = ? WHERE id = ?')
      .run(JobStatus.Completed, Date.now(), id);
  }

  public async fail(id: JobId, _workerId: string, error: string): Promise<void> {
    this.db
      .prepare('UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?')
      .run(JobStatus.Failed, error, Date.now(), id);
  }

  public async list(workspaceId: WorkspaceId): Promise<readonly JobRecord[]> {
    const rows = this.db
      .prepare('SELECT * FROM jobs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT 100')
      .all(workspaceId) as Record<string, unknown>[];
    return rows.map((r) => rowToJob(r, r['status'] as JobStatusValue));
  }

  public async close(): Promise<void> {
    // No-op.
  }
}

const rowToJob = (row: Record<string, unknown>, overrideStatus?: JobStatusValue): JobRecord => {
  const status = (overrideStatus ?? String(row['status'])) as JobStatusValue;
  return {
    id: brandId<JobId>(String(row['id'])),
    workspaceId: brandId<WorkspaceId>(String(row['workspace_id'])),
    ownerId: brandId<UserId>(String(row['owner_id'])),
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

void VectorStoreError;
