/**
 * Background job worker — drains SqliteJobQueue in the foreground.
 *
 * The API server runs one Worker per process. The worker polls
 * `ingestion_jobs` for pending rows, executes them via the provided
 * `handler`, and marks them complete or failed. Designed to be
 * started by `start()` in the API entrypoint.
 *
 * For multi-process deployments swap for an external queue; for a
 * single-machine server this is enough.
 */

import type { Database } from './db-types.js';

export interface JobRecord {
  readonly id: string;
  readonly workspaceId: string;
  readonly ownerId: string;
  readonly kind: string;
  readonly payload: Readonly<Record<string, unknown>>;
  readonly status: 'pending' | 'running' | 'done' | 'failed';
  readonly error: string | null;
  readonly createdAt: number;
  readonly updatedAt: number;
}

export type JobHandler = (job: JobRecord) => Promise<void>;

export interface WorkerOptions {
  readonly db: Database;
  readonly handler: JobHandler;
  readonly pollIntervalMs?: number;
  readonly batchSize?: number;
}

const DEFAULT_POLL = 500;
const DEFAULT_BATCH = 4;

export class JobWorker {
  private readonly db: Database;
  private readonly handler: JobHandler;
  private readonly pollIntervalMs: number;
  private readonly batchSize: number;
  private running = false;
  private loopPromise: Promise<void> | null = null;

  constructor(opts: WorkerOptions) {
    this.db = opts.db;
    this.handler = opts.handler;
    this.pollIntervalMs = opts.pollIntervalMs ?? DEFAULT_POLL;
    this.batchSize = opts.batchSize ?? DEFAULT_BATCH;
  }

  public start(): void {
    if (this.running) return;
    this.running = true;
    this.loopPromise = this.loop();
  }

  public async stop(): Promise<void> {
    this.running = false;
    if (this.loopPromise) await this.loopPromise;
  }

  private async loop(): Promise<void> {
    while (this.running) {
      try {
        const drained = await this.drain();
        if (!drained) {
          await new Promise((r) => setTimeout(r, this.pollIntervalMs));
        }
      } catch {
        await new Promise((r) => setTimeout(r, this.pollIntervalMs));
      }
    }
  }

  /** Returns true if work was found, false if the queue was empty. */
  public async drain(): Promise<boolean> {
    const pending = this.db
      .prepare(
        `SELECT * FROM ingestion_jobs
         WHERE status = 'pending'
         ORDER BY created_at ASC
         LIMIT ?`,
      )
      .all(this.batchSize) as Array<Record<string, unknown>>;
    if (pending.length === 0) return false;
    for (const row of pending) {
      const job = rowToJob(row);
      this.db
        .prepare(
          `UPDATE ingestion_jobs SET status = 'running', updated_at = ? WHERE id = ?`,
        )
        .run(Date.now(), job.id);
      try {
        await this.handler(job);
        this.db
          .prepare(`UPDATE ingestion_jobs SET status = 'done', updated_at = ? WHERE id = ?`)
          .run(Date.now(), job.id);
      } catch (e) {
        this.db
          .prepare(
            `UPDATE ingestion_jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?`,
          )
          .run(e instanceof Error ? e.message : String(e), Date.now(), job.id);
      }
    }
    return true;
  }
}

const rowToJob = (row: Record<string, unknown>): JobRecord => ({
  id: String(row['id']),
  workspaceId: String(row['workspace_id']),
  ownerId: String(row['owner_id']),
  kind: String(row['kind']),
  payload: JSON.parse(String(row['payload_json'] ?? '{}')) as Record<string, unknown>,
  status: String(row['status']) as JobRecord['status'],
  error: row['error'] === null ? null : String(row['error']),
  createdAt: Number(row['created_at']),
  updatedAt: Number(row['updated_at']),
});