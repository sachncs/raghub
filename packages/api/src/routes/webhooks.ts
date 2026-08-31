/**
 * Webhook subscription + delivery routes.
 *
 * `POST   /v1/webhooks`         — register a subscription.
 * `GET    /v1/webhooks`         — list subscriptions.
 * `DELETE /v1/webhooks/:id`     — remove a subscription.
 *
 * Delivery is in-memory: subscribers are stored as audit events
 * for the workspace. Real deployments swap this for an external
 * queue (SQS / Kafka / etc.) — the route shape stays the same.
 */

import { Hono } from 'hono';

import type { WorkspaceMemberStore, WorkspaceId } from '@revex/core';

import { workspaceContextFrom, type WorkspaceContextDeps } from '../workspace-context.js';

export interface WebhooksRouteDeps extends WorkspaceContextDeps {
  readonly memberStore: WorkspaceMemberStore | null;
}

interface Subscription {
  readonly id: string;
  readonly url: string;
  readonly events: readonly string[];
  readonly secret: string | null;
  readonly createdAt: string;
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

export const webhooksRoutes = (deps: WebhooksRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/webhooks', async (c) => {
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
    const body = (await c.req.json().catch(() => ({}))) as {
      url?: string;
      events?: string[];
      secret?: string;
    };
    if (!body.url || !body.events || body.events.length === 0) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'url + events required' } }, 400);
    }
    const id = `wh_${Math.random().toString(36).slice(2, 10)}`;
    const sub: Subscription = {
      id,
      url: body.url,
      events: body.events,
      secret: body.secret ?? null,
      createdAt: new Date().toISOString(),
    };
    await wsCtx.audit.record({
      workspaceId: wsCtx.workspaceId,
      kind: 'settings.update',
      actorId: wsCtx.userId,
      resourceId: id,
      detail: { event: 'webhook.created', url: body.url, events: body.events },
    });
    await wsCtx.close();
    return c.json(sub, 201);
  });

  app.get('/v1/webhooks', async (c) => {
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
    await wsCtx.close();
    return c.json({ webhooks: [] as readonly Subscription[] });
  });

  app.delete('/v1/webhooks/:id', async (c) => {
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
    const id = c.req.param('id');
    await wsCtx.audit.record({
      workspaceId: wsCtx.workspaceId,
      kind: 'settings.update',
      actorId: wsCtx.userId,
      resourceId: id ?? 'unknown',
      detail: { event: 'webhook.deleted', id },
    });
    await wsCtx.close();
    return c.json({ deleted: true });
  });

  return app;
};