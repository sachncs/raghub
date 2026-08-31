/**
 * In-memory job queue + queue base registry.
 *
 * Production code uses `SqliteJobQueue` (cross-process safe). For
 * single-process tests and the CLI's `--no-watch` mode, the
 * in-memory queue is enough.
 *
 * `QueueBase` is the polymorphic base class for queue adapters
 * — register concrete adapters via the `Registry`.
 */

import { ConfigurationError } from './errors/index.js';
import type { JobRecord, JobStatusValue } from './storage/jobs.js';

export type { JobRecord, JobStatusValue };

export interface JobQueue {
  enqueue(job: Omit<JobRecord, 'id' | 'createdAt' | 'updatedAt' | 'status'>): Promise<JobRecord>;
  dequeue(): Promise<JobRecord | null>;
  markRunning(id: string): Promise<void>;
  markDone(id: string): Promise<void>;
  markFailed(id: string, error: string): Promise<void>;
  list(opts?: { status?: JobStatusValue; limit?: number }): Promise<readonly JobRecord[]>;
  resetStuckJobs(): Promise<number>;
  close(): Promise<void>;
  size(): Promise<number>;
}

export const JobStatus = {
  Pending: 'pending',
  Running: 'running',
  Completed: 'completed',
  Failed: 'failed',
} as const;

export class MemoryQueue implements JobQueue {
  private readonly pending: JobRecord[] = [];
  private readonly all = new Map<string, JobRecord>();
  private seq = 0;

  async enqueue(input: Omit<JobRecord, 'id' | 'createdAt' | 'updatedAt' | 'status'>): Promise<JobRecord> {
    const id = `mem_${++this.seq}`;
    const now = new Date();
    const rec: JobRecord = { ...input, id: id as never, status: JobStatus.Pending, createdAt: now, updatedAt: now };
    this.pending.push(rec);
    this.all.set(id, rec);
    return rec;
  }

  async dequeue(): Promise<JobRecord | null> {
    const next = this.pending.shift() ?? null;
    if (next) {
      const updated: JobRecord = { ...next, status: JobStatus.Running, updatedAt: new Date() };
      this.all.set(updated.id, updated);
      return updated;
    }
    return null;
  }

  async markRunning(id: string): Promise<void> {
    const rec = this.all.get(id);
    if (!rec) return;
    this.all.set(id, { ...rec, status: JobStatus.Running, updatedAt: new Date() });
  }

  async markDone(id: string): Promise<void> {
    const rec = this.all.get(id);
    if (!rec) return;
    this.all.set(id, { ...rec, status: JobStatus.Completed, updatedAt: new Date() });
  }

  async markFailed(id: string, error: string): Promise<void> {
    const rec = this.all.get(id);
    if (!rec) return;
    this.all.set(id, { ...rec, status: JobStatus.Failed, error, updatedAt: new Date() });
  }

  async list(opts?: { status?: JobStatusValue; limit?: number }): Promise<readonly JobRecord[]> {
    let out = Array.from(this.all.values());
    if (opts?.status) out = out.filter((r) => r.status === opts.status);
    out.sort((a, b) => +b.createdAt - +a.createdAt);
    if (opts?.limit) out = out.slice(0, opts.limit);
    return out;
  }

  async resetStuckJobs(): Promise<number> {
    let n = 0;
    for (const [id, rec] of this.all) {
      if (rec.status === JobStatus.Running) {
        this.all.set(id, { ...rec, status: JobStatus.Pending, updatedAt: new Date() });
        this.pending.push(rec);
        n += 1;
      }
    }
    return n;
  }

  async close(): Promise<void> {
    this.pending.length = 0;
    this.all.clear();
  }

  async size(): Promise<number> {
    return this.all.size;
  }
}

export class QueueBaseRegistry {
  private static readonly map = new Map<string, JobQueue>();

  static register(name: string, queue: JobQueue): void {
    if (QueueBaseRegistry.map.has(name)) {
      throw new ConfigurationError(`queue adapter already registered: ${name}`);
    }
    QueueBaseRegistry.map.set(name, queue);
  }

  static require(name: string): JobQueue {
    const q = QueueBaseRegistry.map.get(name);
    if (!q) throw new ConfigurationError(`unknown queue adapter: ${name}`);
    return q;
  }

  static names(): readonly string[] {
    return Array.from(QueueBaseRegistry.map.keys());
  }
}