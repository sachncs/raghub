/**
 * Audit timeline + workspace stats routes.
 *
 * `GET /v1/audit`         — paginated audit events (admin/owner).
 * `GET /v1/audit/kinds`   — distinct event kinds.
 * `GET /v1/stats`         — workspace counters.
 */

import { Hono } from 'hono';

import type { WorkspaceMemberStore, WorkspaceId } from '@revex/core';

import { workspaceContextFrom, type WorkspaceContextDeps } from '../workspace-context.js';

export interface AuditRouteDeps extends WorkspaceContextDeps {
  readonly memberStore: WorkspaceMemberStore | null;
}

const isAdmin = async (
  memberStore: WorkspaceMemberStore | null,
  workspaceId: WorkspaceId,
  userId: string,
): Promise<boolean> => {
  if (!memberStore) return true;
  const member = await memberStore.get(workspaceId, userId as never);
  return member?.role === 'admin' || member?.role === 'owner';
};

export const auditRoutes = (deps: AuditRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/audit', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    if (!(await isAdmin(deps.memberStore, wsCtx.workspaceId, wsCtx.userId))) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'forbidden' } }, 403);
    }
    const limit = Number(c.req.query('limit') ?? '100');
    const kind = c.req.query('kind');
    const all = await wsCtx.audit.list({
      workspaceId: wsCtx.workspaceId,
      limit: Math.min(Math.max(limit, 1), 500),
    });
    await wsCtx.close();
    const events = kind ? all.filter((e) => e.kind === kind) : all;
    return c.json({ events });
  });

  app.get('/v1/audit/kinds', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    if (!(await isAdmin(deps.memberStore, wsCtx.workspaceId, wsCtx.userId))) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'forbidden' } }, 403);
    }
    const events = await wsCtx.audit.list({
      workspaceId: wsCtx.workspaceId,
      limit: 1000,
    });
    await wsCtx.close();
    const kinds = new Set<string>();
    for (const e of events) kinds.add(e.kind);
    return c.json({ kinds: Array.from(kinds).sort() });
  });

  app.get('/v1/stats', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    const docs = await wsCtx.documentStore.listForUser(
      wsCtx.workspaceId,
      wsCtx.userId,
    );
    const docsByStatus = new Map<string, number>();
    for (const d of docs) docsByStatus.set(d.status, (docsByStatus.get(d.status) ?? 0) + 1);
    const auditCount = (await wsCtx.audit.list({
      workspaceId: wsCtx.workspaceId,
      limit: 100000,
    })).length;
    const feedback = wsCtx.feedback.aggregate(wsCtx.workspaceId);
    await wsCtx.close();
    return c.json({
      documents: { total: docs.length, byStatus: Object.fromEntries(docsByStatus) },
      audit: { events: auditCount },
      feedback,
    });
  });

  return app;
};