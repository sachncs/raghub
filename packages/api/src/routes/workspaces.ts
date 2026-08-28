/**
 * Workspace routes — member management + invite flow.
 *
 * POST /v1/workspaces/members          invite a user (admin only)
 * GET  /v1/workspaces/members          list members
 * PATCH /v1/workspaces/members/:userId change role
 * DELETE /v1/workspaces/members/:userId remove a member
 *
 * The first member (owner) is created at workspace registration
 * time. This route handles subsequent invites.
 */

import { Hono } from 'hono';

import {
  brandId,
  canManageWorkspace,
  type Embedder,
  type SqliteAuditEventStore,
  type UserId,
  type VectorStore,
  type WorkspaceId,
  type WorkspaceMemberStore,
  type WorkspaceMemberRoleValue,
} from '@raghub/core';

import { getClaims } from '../middleware/auth.js';
import { workspaceContextFrom } from '../workspace-context.js';

export interface WorkspaceRouteDeps {
  readonly pool: import('../workspace-pool.js').WorkspacePool;
  readonly memberStore: WorkspaceMemberStore | null;
  readonly audit: SqliteAuditEventStore | null;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore | null;
}

const allRoles: readonly WorkspaceMemberRoleValue[] = ['owner', 'admin', 'member', 'viewer'];

const isRole = (s: string): s is WorkspaceMemberRoleValue =>
  (allRoles as readonly string[]).includes(s);

export const workspaceRoutes = (deps: WorkspaceRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/workspaces/members', async (c) => {
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const memberStore = ctx.memberStore;
    const workspaceId = ctx.workspaceId;
    const list = await memberStore.list(workspaceId);
    return c.json({
      members: list.map((m) => ({
        userId: m.userId,
        role: m.role,
        joinedAt: m.joinedAt.toISOString(),
      })),
    });
  });

  app.post('/v1/workspaces/members', async (c) => {
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const memberStore = ctx.memberStore;
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const audit = deps.audit;
    const me = await memberStore.get(workspaceId, userId);
    if (!me || !canManageWorkspace(me.role)) {
      return c.json({ error: { code: 'authorization_error', message: 'only admins can invite' } }, 403);
    }
    const body = (await c.req.json().catch(() => ({}))) as { email?: string; role?: string };
    if (!body.email || !body.role || !isRole(body.role)) {
      return c.json({ error: { code: 'raghub_error', message: 'email + role required' } }, 400);
    }
    const newUserId = brandId<UserId>(`usr_${body.email.split('@')[0]}_${Math.random().toString(36).slice(2, 8)}`);
    const member = await memberStore.upsert({
      workspaceId,
      userId: newUserId,
      role: body.role,
    });
    if (deps.audit) {
      await deps.audit.record({
        kind: 'workspace.member.add',
        workspaceId,
        actorId: brandId<UserId>(userId),
        resourceId: newUserId,
        detail: { email: body.email, role: body.role },
      });
    }
    return c.json({ member }, 201);
  });

  app.patch('/v1/workspaces/members/:userId', async (c) => {
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const memberStore = ctx.memberStore;
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const audit = deps.audit;
    const me = await memberStore.get(workspaceId, userId);
    if (!me || !canManageWorkspace(me.role)) {
      return c.json({ error: { code: 'authorization_error', message: 'only admins can change roles' } }, 403);
    }
    const body = (await c.req.json().catch(() => ({}))) as { role?: string };
    if (!body.role || !isRole(body.role)) {
      return c.json({ error: { code: 'raghub_error', message: 'role required' } }, 400);
    }
    const target = brandId<UserId>(c.req.param('userId'));
    const updated = await memberStore.upsert({
      workspaceId,
      userId: target,
      role: body.role,
    });
    if (deps.audit) {
      await deps.audit.record({
        kind: 'workspace.member.role_change',
        workspaceId,
        actorId: brandId<UserId>(userId),
        resourceId: target,
        detail: { role: body.role },
      });
    }
    return c.json({ member: updated });
  });

  app.delete('/v1/workspaces/members/:userId', async (c) => {
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const memberStore = ctx.memberStore;
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const audit = deps.audit;
    const me = await memberStore.get(workspaceId, userId);
    if (!me || !canManageWorkspace(me.role)) {
      return c.json({ error: { code: 'authorization_error', message: 'only admins can remove' } }, 403);
    }
    const target = brandId<UserId>(c.req.param('userId'));
    await memberStore.remove(workspaceId, target);
    if (deps.audit) {
      await deps.audit.record({
        kind: 'workspace.member.remove',
        workspaceId,
        actorId: brandId<UserId>(userId),
        resourceId: target,
        detail: {},
      });
    }
    return c.json({ ok: true });
  });

  return app;
};