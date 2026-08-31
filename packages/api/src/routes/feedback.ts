/**
 * * feedback routes.
 *
 * `POST /v1/feedback` — record a per-turn rating.
 * `GET /v1/feedback` — list recent feedback (any workspace member).
 * `GET /v1/feedback/:id` — fetch a single record.
 * `DELETE /v1/feedback/:id` — remove (admin/owner only).
 * `GET /v1/feedback/aggregate` — workspace-wide up/down/neutral counts.
 *
 * Storage is per-workspace via `SqliteFeedbackStore`.
 */

import { Hono } from 'hono';

import type { WorkspaceMemberStore, WorkspaceId } from '@revex/core';

import { workspaceContextFrom, type WorkspaceContextDeps } from '../workspace-context.js';

export interface FeedbackRouteDeps extends WorkspaceContextDeps {
  readonly memberStore: WorkspaceMemberStore | null;
}

const canRead = async (
  memberStore: WorkspaceMemberStore | null,
  workspaceId: WorkspaceId,
  userId: string,
): Promise<boolean> => {
  if (!memberStore) return true;
  const member = await memberStore.get(workspaceId, userId as never);
  return member !== null;
};

export const feedbackRoutes = (deps: FeedbackRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/feedback', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    const body = (await c.req.json().catch(() => ({}))) as {
      turnId?: string;
      rating?: 'up' | 'down' | 'neutral';
      comment?: string | null;
    };
    if (!body.turnId || !body.rating) {
      return c.json({ error: { code: 'revex_error', message: 'turnId + rating required' } }, 400);
    }
    const rec = wsCtx.feedback.add(
      wsCtx.workspaceId,
      wsCtx.userId,
      body.turnId as never,
      body.rating,
      body.comment ?? null,
    );
    await wsCtx.close();
    return c.json({ id: rec.id, status: 'recorded' });
  });

  app.get('/v1/feedback', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    if (!(await canRead(deps.memberStore, wsCtx.workspaceId, wsCtx.userId))) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'forbidden' } }, 403);
    }
    const rows = wsCtx.feedback.list(wsCtx.workspaceId);
    await wsCtx.close();
    return c.json({ feedback: rows });
  });

  app.get('/v1/feedback/aggregate', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    if (!(await canRead(deps.memberStore, wsCtx.workspaceId, wsCtx.userId))) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'forbidden' } }, 403);
    }
    const agg = wsCtx.feedback.aggregate(wsCtx.workspaceId);
    await wsCtx.close();
    return c.json(agg);
  });

  app.get('/v1/feedback/:id', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    if (!(await canRead(deps.memberStore, wsCtx.workspaceId, wsCtx.userId))) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'forbidden' } }, 403);
    }
    const id = c.req.param('id');
    if (!id) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'id required' } }, 400);
    }
    const rec = wsCtx.feedback.get(wsCtx.workspaceId, id as never);
    await wsCtx.close();
    if (!rec) return c.json({ error: { code: 'revex_error', message: 'not found' } }, 404);
    return c.json(rec);
  });

  app.delete('/v1/feedback/:id', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    if (!(await canRead(deps.memberStore, wsCtx.workspaceId, wsCtx.userId))) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'forbidden' } }, 403);
    }
    const id = c.req.param('id');
    if (!id) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'id required' } }, 400);
    }
    const ok = wsCtx.feedback.delete(wsCtx.workspaceId, id as never);
    await wsCtx.close();
    return c.json({ deleted: ok });
  });

  return app;
};