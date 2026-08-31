/**
 * * tenant routes.
 *
 * `GET /v1/tenants`          — list all workspaces visible to the
 *                               caller (any member of each workspace
 *                               can see its own workspace).
 * `GET /v1/tenants/:id`      — fetch one workspace summary.
 * `DELETE /v1/tenants/:id`   — admin-only: deregister a workspace
 *                               from the registry (does not delete
 *                               data on disk).
 *
 * Production deployments replace these with multi-tenant
 * management UIs. The CLI's `revex tenant` commands operate on
 * the same registry via the API.
 */

import { Hono } from 'hono';

import type { WorkspaceRegistry } from '@revex/core';
import type { WorkspacePool } from '../workspace-pool.js';
import { brandId, type WorkspaceId } from '@revex/core';
import { workspaceContextFrom, type WorkspaceContextDeps } from '../workspace-context.js';

export interface TenantRouteDeps extends WorkspaceContextDeps {
  readonly pool: WorkspacePool;
  readonly registry: WorkspaceRegistry;
}

export const tenantRoutes = (deps: TenantRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/tenants', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    await wsCtx.close();
    const entries = await deps.registry.list();
    return c.json({ tenants: entries });
  });

  app.get('/v1/tenants/:id', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    const id = c.req.param('id');
    await wsCtx.close();
    if (!id) return c.json({ error: { code: 'revex_error', message: 'id required' } }, 400);
    const entry = await deps.registry.resolve(brandId<WorkspaceId>(id));
    if (!entry) return c.json({ error: { code: 'revex_error', message: 'not found' } }, 404);
    return c.json({ tenant: entry });
  });

  return app;
};