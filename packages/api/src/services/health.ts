/**
 * Per-collaborator health aggregator.
 *
 * Probes each registered collaborator (vector store, embedder,
 * queue, registry) and returns an aggregate HealthReport. The
 * `/readyz` route uses this for liveness checks.
 */

import type { Embedder, VectorStore, WorkspaceRegistry, SqliteJobQueue } from '@revex/core';

export type HealthStatus = 'ok' | 'degraded' | 'down';

export interface ComponentHealth {
  readonly status: HealthStatus;
  readonly latencyMs: number;
  readonly detail: Readonly<Record<string, unknown>>;
}

export interface HealthReport {
  readonly status: HealthStatus;
  readonly components: Readonly<Record<string, ComponentHealth>>;
  readonly checkedAt: string;
}

export interface HealthDeps {
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore | null;
  readonly jobQueue: SqliteJobQueue | null;
  readonly registry: WorkspaceRegistry;
}

export const probeHealth = async (deps: HealthDeps): Promise<HealthReport> => {
  const components: Record<string, ComponentHealth> = {};

  const embedderProbe = await time(async () => {
    await deps.embedder.embedQuery('health-check');
  });
  components.embedder = {
    status: embedderProbe.ok ? 'ok' : 'down',
    latencyMs: embedderProbe.ms,
    detail: {},
  };

  if (deps.vectorStore) {
    const vecProbe = await time(async () => {
      // VectorStore interface does not expose a health() method;
      // probe by running a no-op keyword search.
      await deps.vectorStore?.searchKeyword({
        query: '__probe__',
        topK: 1,
        filter: {
          workspaceId: '__probe__' as never,
          userId: null,
          collectionId: null,
          principals: [],
          allowedCompanies: [],
        },
      });
    });
    components.vectorstore = {
      status: vecProbe.ok ? 'ok' : 'down',
      latencyMs: vecProbe.ms,
      detail: {},
    };
  }

  if (deps.jobQueue) {
    const queueProbe = await time(async () => {
      await deps.jobQueue?.list('__health__' as never);
    });
    components.queue = {
      status: queueProbe.ok ? 'ok' : 'down',
      latencyMs: queueProbe.ms,
      detail: {},
    };
  }

  const registryProbe = await time(async () => {
    await deps.registry.list();
  });
  components.registry = {
    status: registryProbe.ok ? 'ok' : 'down',
    latencyMs: registryProbe.ms,
    detail: {},
  };

  const statuses = Object.values(components).map((c) => c.status);
  const aggregate: HealthStatus = statuses.every((s) => s === 'ok')
    ? 'ok'
    : statuses.some((s) => s === 'down')
      ? 'down'
      : 'degraded';

  return {
    status: aggregate,
    components,
    checkedAt: new Date().toISOString(),
  };
};

const time = async (
  fn: () => Promise<unknown>,
): Promise<{ ok: boolean; ms: number }> => {
  const start = Date.now();
  try {
    await fn();
    return { ok: true, ms: Date.now() - start };
  } catch {
    return { ok: false, ms: Date.now() - start };
  }
};