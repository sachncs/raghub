/**
 * JWT auth middleware.
 *
 * Reads the bearer token from `Authorization: Bearer <jwt>`, verifies
 * it via the bound `JwtService`, and stores the decoded claims on
 * `c.var.claims`. Tenant binding happens downstream in the route
 * handlers (they look up the user from the UserStore).
 */

import type { MiddlewareHandler, Context } from 'hono';

import { type JwtClaims, type JwtService } from '@raghub/core';

export interface AuthVars {
  readonly claims: JwtClaims;
}

export const jwtAuthMiddleware = (jwt: JwtService): MiddlewareHandler => async (c, next) => {
  const header = c.req.header('authorization') ?? '';
  const m = /^Bearer\s+(.+)$/i.exec(header);
  if (!m || !m[1]) {
    return c.json({ error: { code: 'auth_error', message: 'missing bearer token' } }, 401);
  }
  try {
    const claims = await jwt.verify(m[1]);
    c.set('claims', claims);
    await next();
  } catch (e) {
    return c.json(
      {
        error: {
          code: 'auth_error',
          message: e instanceof Error ? e.message : 'invalid token',
        },
      },
      401,
    );
  }
};

export const getClaims = (c: Context): JwtClaims => {
  const claims = c.get('claims');
  if (!claims) throw new Error('claims missing; jwtAuthMiddleware not applied?');
  return claims;
};