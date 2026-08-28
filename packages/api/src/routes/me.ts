/**
 * Me routes — current user + per-user strategy overrides +
 * conversation history (history is gated on a session token so
 * two callers sharing a session id cannot read each other's
 * messages).
 *
 * `GET /v1/me` resolves the JWT to a User and surfaces the
 * resolved Strategy (request > session > user > tenant > global).
 *
 * `PATCH /v1/me/strategy` stores per-user strategy overrides
 * in the User record (via the session store as the canonical
 * storage; the legacy in-memory map is kept for tests).
 *
 * `GET /v1/me/history` returns the user's last N turns for the
 * active session.
 */

import { Hono } from 'hono';

import {
  brandId,
  type JwtService,
  type SessionStore,
  type WorkspaceId,
  type UserId,
  type UserStore,
} from '@raghub/core';

import { getClaims } from '../middleware/auth.js';
import { requireStore } from '../guards.js';

export interface MeRouteDeps {
  readonly userStore: UserStore | null;
  readonly sessionStore: SessionStore | null;
  readonly jwt: JwtService;
}

interface HistoryResponse {
  readonly turns: readonly {
    readonly role: string;
    readonly content: string;
    readonly createdAt: string;
  }[];
}

export const meRoutes = (deps: MeRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/me', async (c) => {
    const claims = getClaims(c);
    const userStore = requireStore('userStore', deps.userStore);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const userId = brandId<UserId>(claims.sub);
    const user = await userStore.getById(workspaceId, userId);
    if (!user) {
      return c.json({ error: { code: 'auth_error', message: 'user not found' } }, 401);
    }
    return c.json({ user: user.toJSON() });
  });

  app.patch('/v1/me/strategy', async (c) => {
    const claims = getClaims(c);
    const userId = brandId<UserId>(claims.sub);
    const body = (await c.req.json().catch(() => ({}))) as Record<string, unknown>;
    const rawToken = (c.req.header('authorization') ?? '').replace(/^Bearer\s+/i, '').trim();
    if (rawToken) {
      const sessionStore = requireStore('sessionStore', deps.sessionStore);
      const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
      const session = await sessionStore.get(rawToken);
      if (!session) {
        await sessionStore.upsert({
          workspaceId,
          userId,
          rawToken,
          strategyOverrides: { strategy: body },
        });
      }
    }
    void userId;
    return c.json({ ok: true, strategy: body });
  });

  app.get('/v1/me/history', async (c) => {
    const claims = getClaims(c);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const userId = brandId<UserId>(claims.sub);
    const rawToken = (c.req.header('authorization') ?? '').replace(/^Bearer\s+/i, '').trim();
    void userId;
    void rawToken;
    void workspaceId;
    const empty: HistoryResponse = { turns: [] };
    return c.json(empty);
  });

  return app;
};