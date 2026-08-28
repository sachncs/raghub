/**
 * WorkspaceWorkerSupervisor — manages JobWorker instances per
 * registered workspace.
 *
 * Polls `workspaceRegistry.value` (a Set<workspaceId> mutated by
 * auth/register) every pollMs and ensures a JobWorker is running
 * for every registered workspace. For multi-tenant production
 * the poll cadence should be replaced with a push notification;
 * for now a 2s poll is fine.
 */

import type { Database } from './db-types.js';
import { JobWorker, type JobHandler } from './job-worker.js';
import { workspaceRegistry } from './workspace-vault.js';

export interface WorkspaceSupervisorDeps {
  readonly resolveDb: (workspaceId: string) => Promise<Database | null> | Database | null;
  readonly resolveHandler: (workspaceId: string) => JobHandler | null;
  readonly pollMs?: number;
}

export class WorkspaceWorkerSupervisor {
  private readonly resolveDb: (workspaceId: string) => Promise<Database | null> | Database | null;
  private readonly resolveHandler: (workspaceId: string) => JobHandler | null;
  private readonly pollMs: number;
  private readonly workers = new Map<string, JobWorker>();
  private pollHandle: NodeJS.Timeout | null = null;

  constructor(deps: WorkspaceSupervisorDeps) {
    this.resolveDb = deps.resolveDb;
    this.resolveHandler = deps.resolveHandler;
    this.pollMs = deps.pollMs ?? 2_000;
  }

  public start(): void {
    if (this.pollHandle) return;
    this.pollHandle = setInterval(() => {
      void this.scan();
    }, this.pollMs);
    void this.scan();
  }

  public async stop(): Promise<void> {
    if (this.pollHandle) {
      clearInterval(this.pollHandle);
      this.pollHandle = null;
    }
    for (const worker of this.workers.values()) {
      await worker.stop();
    }
    this.workers.clear();
  }

  private async scan(): Promise<void> {
    for (const id of workspaceRegistry.value) {
      if (this.workers.has(id)) continue;
      const db = await this.resolveDb(id);
      const handler = this.resolveHandler(id);
      if (!db || !handler) {
        if (process.env['RAGHUB_DEBUG_SUPERVISOR']) {
          // eslint-disable-next-line no-console
          console.log(`[supervisor] skip ${id}: db=${!!db} handler=${!!handler}`);
        }
        continue;
      }
      const worker = new JobWorker({ db, handler });
      worker.start();
      this.workers.set(id, worker);
      // eslint-disable-next-line no-console
      console.log(`[supervisor] started worker for ${id}`);
    }
  }
}