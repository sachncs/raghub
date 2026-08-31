/**
 * Operational routes — health, ready, and a tiny diagnostics dump.
 *
 * `/health`  — liveness (process alive)
 * `/readyz`  — readiness (registry reachable, no active lock)
 * `/v1/diagnostics` — auth-required; surfaces workspace info
 *
 * These never leak secrets or schema details.
 */

import { Hono } from 'hono';

import type { WorkspaceRegistry } from '@revex/core';

import { getClaims } from '../middleware/auth.js';
import type { WorkspacePool } from '../workspace-pool.js';

export interface OperationalRouteDeps {
  readonly registry: WorkspaceRegistry;
  readonly pool: WorkspacePool;
  readonly startTime: number;
  readonly version: string;
}

export const operationalRoutes = (deps: OperationalRouteDeps): Hono => {
  const app = new Hono();

  app.get('/health', (c) => c.json({ ok: true, version: deps.version }));

  app.get('/readyz', async (c) => {
    try {
      const list = await deps.registry.list();
      return c.json({ ok: true, workspaces: list.length, uptimeSec: Math.floor((Date.now() - deps.startTime) / 1000) });
    } catch (e) {
      return c.json(
        { ok: false, error: e instanceof Error ? e.message : String(e) },
        503,
      );
    }
  });

  app.get('/v1/diagnostics', async (c) => {
    const claims = getClaims(c);
    const entry = await deps.registry.resolve(claims.workspace_id as never);
    return c.json({
      ok: true,
      workspace: entry
        ? { id: entry.workspaceId, encryption: entry.encryption, registeredAt: entry.registeredAt.toISOString() }
        : null,
      poolSize: deps.pool.size(),
      uptimeSec: Math.floor((Date.now() - deps.startTime) / 1000),
    });
  });

  return app;
};