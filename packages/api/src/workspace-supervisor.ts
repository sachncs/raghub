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
import type { WorkspaceRegistry } from '@raghub/core';
import { openEncryptedWorkspace } from '@raghub/core';

import { workspaceRegistry, passVaultRef } from './workspace-bootstrap.js';

export interface WorkspaceSupervisorDeps {
  readonly registry: WorkspaceRegistry;
  readonly resolveHandler: (workspaceId: string) => JobHandler | null;
  readonly pollMs?: number;
}

export class WorkspaceWorkerSupervisor {
  private readonly registry: WorkspaceRegistry;
  private readonly resolveHandler: (workspaceId: string) => JobHandler | null;
  private readonly pollMs: number;
  private readonly workers = new Map<string, JobWorker>();
  private pollHandle: NodeJS.Timeout | null = null;

  constructor(deps: WorkspaceSupervisorDeps) {
    this.registry = deps.registry;
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
      if (!db || !handler) continue;
      const worker = new JobWorker({ db, handler });
      worker.start();
      this.workers.set(id, worker);
    }
  }

  /**
   * resolveDb — opens the workspace via the registry + the
   * configured passphrase vault. The vault interface means
   * production can swap in KMS without changing this class.
   */
  private async resolveDb(workspaceId: string): Promise<Database | null> {
    const entry = await this.registry.resolve(workspaceId as never);
    const vault = passVaultRef.value;
    if (!entry || !vault) return null;
    const passphrase = await vault.get(workspaceId);
    if (passphrase === null) return null;
    try {
      const handle = await openEncryptedWorkspace({
        path: entry.path,
        passphrase,
      });
      return handle.db as never;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        `[supervisor] failed to open workspace ${workspaceId}: ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      return null;
    }
  }
}