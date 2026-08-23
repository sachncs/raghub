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
  type UserId,
  type WorkspaceId,
  type WorkspaceMemberStore,
  type WorkspaceMemberRoleValue,
} from '@raghub/core';

import { getClaims } from '../middleware/auth.js';

export interface WorkspaceRouteDeps {
  readonly memberStore: WorkspaceMemberStore;
}

const allRoles: readonly WorkspaceMemberRoleValue[] = ['owner', 'admin', 'member', 'viewer'];

const isRole = (s: string): s is WorkspaceMemberRoleValue =>
  (allRoles as readonly string[]).includes(s);

export const workspaceRoutes = (deps: WorkspaceRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/workspaces/members', async (c) => {
    const claims = getClaims(c);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const list = await deps.memberStore.list(workspaceId);
    return c.json({
      members: list.map((m) => ({
        userId: m.userId,
        role: m.role,
        joinedAt: m.joinedAt.toISOString(),
      })),
    });
  });

  app.post('/v1/workspaces/members', async (c) => {
    const claims = getClaims(c);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const me = await deps.memberStore.get(workspaceId, brandId<UserId>(claims.sub));
    if (!me || !canManageWorkspace(me.role)) {
      return c.json({ error: { code: 'authorization_error', message: 'only admins can invite' } }, 403);
    }
    const body = (await c.req.json().catch(() => ({}))) as { email?: string; role?: string };
    if (!body.email || !body.role || !isRole(body.role)) {
      return c.json({ error: { code: 'raghub_error', message: 'email + role required' } }, 400);
    }
    const newUserId = brandId<UserId>(`usr_${body.email.split('@')[0]}_${Math.random().toString(36).slice(2, 8)}`);
    const member = await deps.memberStore.upsert({
      workspaceId,
      userId: newUserId,
      role: body.role,
    });
    return c.json({ member }, 201);
  });

  app.patch('/v1/workspaces/members/:userId', async (c) => {
    const claims = getClaims(c);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const me = await deps.memberStore.get(workspaceId, brandId<UserId>(claims.sub));
    if (!me || !canManageWorkspace(me.role)) {
      return c.json({ error: { code: 'authorization_error', message: 'only admins can change roles' } }, 403);
    }
    const body = (await c.req.json().catch(() => ({}))) as { role?: string };
    if (!body.role || !isRole(body.role)) {
      return c.json({ error: { code: 'raghub_error', message: 'role required' } }, 400);
    }
    const target = brandId<UserId>(c.req.param('userId'));
    const updated = await deps.memberStore.upsert({
      workspaceId,
      userId: target,
      role: body.role,
    });
    return c.json({ member: updated });
  });

  app.delete('/v1/workspaces/members/:userId', async (c) => {
    const claims = getClaims(c);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const me = await deps.memberStore.get(workspaceId, brandId<UserId>(claims.sub));
    if (!me || !canManageWorkspace(me.role)) {
      return c.json({ error: { code: 'authorization_error', message: 'only admins can remove' } }, 403);
    }
    const target = brandId<UserId>(c.req.param('userId'));
    await deps.memberStore.remove(workspaceId, target);
    return c.json({ ok: true });
  });

  return app;
};