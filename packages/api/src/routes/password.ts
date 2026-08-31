/**
 * Password reset route.
 *
 * `POST /v1/auth/reset` — admin-only password reset for another
 * member of the workspace.
 *
 * `POST /v1/auth/password` — self-service password change. The
 * caller supplies the current + new password.
 */

import { Hono } from 'hono';

import type {
  BcryptHasher,
  SqliteUserStore,
  WorkspaceMemberStore,
  WorkspaceId,
} from '@revex/core';

import { workspaceContextFrom, type WorkspaceContextDeps } from '../workspace-context.js';
import { getClaims } from '../middleware/auth.js';

export interface PasswordRouteDeps extends WorkspaceContextDeps {
  readonly userStore: SqliteUserStore | null;
  readonly memberStore: WorkspaceMemberStore | null;
  readonly hasher: BcryptHasher;
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

export const passwordRoutes = (deps: PasswordRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/auth/password', async (c) => {
    let wsCtx: Awaited<ReturnType<typeof workspaceContextFrom>>;
    try {
      wsCtx = await workspaceContextFrom(c, deps);
    } catch {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    const body = (await c.req.json().catch(() => ({}))) as {
      currentPassword?: string;
      newPassword?: string;
    };
    if (!body.currentPassword || !body.newPassword) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'currentPassword + newPassword required' } }, 400);
    }
    if (body.newPassword.length < 8) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'newPassword must be >= 8 chars' } }, 400);
    }
    if (!deps.userStore) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'user store unavailable' } }, 503);
    }
    const claims = getClaims(c as never);
    const userId = (claims.sub ?? '') as never;
    const lookup = await deps.userStore.getById(wsCtx.workspaceId, userId);
    if (!lookup) {
      await wsCtx.close();
      return c.json({ error: { code: 'auth_error', message: 'unknown user' } }, 401);
    }
    const ok = await deps.hasher.verify(body.currentPassword, lookup.passwordHash);
    if (!ok) {
      await wsCtx.close();
      return c.json({ error: { code: 'auth_error', message: 'current password mismatch' } }, 401);
    }
    await deps.userStore.updatePassword(
      wsCtx.workspaceId,
      userId,
      await deps.hasher.hash(body.newPassword),
    );
    await wsCtx.close();
    return c.json({ ok: true });
  });

  app.post('/v1/auth/reset', async (c) => {
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
      userId?: string;
      newPassword?: string;
    };
    if (!body.userId || !body.newPassword) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'userId + newPassword required' } }, 400);
    }
    if (body.newPassword.length < 8) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'newPassword must be >= 8 chars' } }, 400);
    }
    if (!deps.userStore) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'user store unavailable' } }, 503);
    }
    const target = await deps.userStore.getById(wsCtx.workspaceId, body.userId as never);
    if (!target) {
      await wsCtx.close();
      return c.json({ error: { code: 'revex_error', message: 'target user not found' } }, 404);
    }
    await deps.userStore.updatePassword(
      wsCtx.workspaceId,
      body.userId as never,
      await deps.hasher.hash(body.newPassword),
    );
    await wsCtx.audit.record({
      workspaceId: wsCtx.workspaceId,
      kind: 'workspace.member.role_change',
      actorId: wsCtx.userId,
      resourceId: body.userId,
      detail: { event: 'password.reset' },
    });
    await wsCtx.close();
    return c.json({ ok: true });
  });

  return app;
};