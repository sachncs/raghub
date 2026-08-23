/**
 * Me routes — current user + per-user strategy overrides.
 *
 * `GET /v1/me` resolves the JWT to a User (re-checked against the
 * UserStore so a revoked account immediately loses access).
 *
 * `PATCH /v1/me/strategy` stores a per-user strategy override
 * (JSON-encoded on the User row in a follow-up commit; Phase 1
 * accepts the patch and stores it in-memory keyed by user id).
 */

import { Hono } from 'hono';

import { brandId, type JwtService, type TenantId, type UserId, type UserStore } from '@raghub/core';

import { getClaims } from '../middleware/auth.js';

export interface MeRouteDeps {
  readonly userStore: UserStore;
  readonly jwt: JwtService;
}

type StrategyOverrides = Record<string, unknown>;
const memoryStrategyStore = new Map<string, StrategyOverrides>();

export const meRoutes = (deps: MeRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/me', async (c) => {
    const claims = getClaims(c);
    const user = await deps.userStore.getById(
      brandId<TenantId>(claims.tenant_id),
      brandId<UserId>(claims.sub),
    );
    if (!user) {
      return c.json({ error: { code: 'auth_error', message: 'user not found' } }, 401);
    }
    const overrides = memoryStrategyStore.get(user.id) ?? null;
    return c.json({ user: user.toJSON(), strategy: overrides });
  });

  app.patch('/v1/me/strategy', async (c) => {
    const claims = getClaims(c);
    const body = (await c.req.json().catch(() => ({}))) as Partial<StrategyOverrides>;
    const userId = claims.sub;
    memoryStrategyStore.set(userId, body);
    return c.json({ ok: true, strategy: body });
  });

  return app;
};