/**
 * Logout endpoint — invalidates the session row and clears cookies.
 *
 * The JWT itself stays valid until its `exp`; full revocation
 * requires a server-side deny-list keyed by token. Until then,
 * logout clears the session record so the user can't fetch new
 * strategy overrides or upload via the cached session.
 */

import { Hono } from 'hono';

import { type SessionStore, type WorkspaceId } from '@raghub/core';

import { getClaims } from '../middleware/auth.js';
import { requireStore } from '../guards.js';

export interface LogoutRouteDeps {
  readonly sessionStore: SessionStore | null;
}

export const logoutRoutes = (deps: LogoutRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/auth/logout', async (c) => {
    const claims = getClaims(c);
    const token = (c.req.header('authorization') ?? '').replace(/^Bearer\s+/i, '').trim();
    if (token) {
      const sessionStore = requireStore('sessionStore', deps.sessionStore);
      await sessionStore.remove(token, claims.workspace_id as WorkspaceId);
    }
    c.header('Set-Cookie', 'raghub_token=; path=/; max-age=0; samesite=lax');
    c.header('Set-Cookie', 'raghub_passphrase=; path=/; max-age=0; samesite=lax');
    return c.json({ ok: true });
  });

  return app;
};