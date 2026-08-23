/**
 * JWT auth middleware.
 *
 * Reads the bearer token from `Authorization: Bearer <jwt>`, verifies
 * it via the bound `JwtService`, and stores the decoded claims on
 * `c.var.claims`. The cookie `raghub_passphrase` is also extracted
 * (the browser sends it on every request so the server can decrypt
 * the workspace's settings table).
 *
 * Workspace binding happens downstream in the route handlers (they
 * look up the user from the UserStore + grab the handle from the
 * WorkspacePool).
 */

import type { Context, MiddlewareHandler } from 'hono';

import { type JwtClaims, type JwtService } from '@raghub/core';

export interface AuthVars {
  readonly claims: JwtClaims;
  readonly passphrase: string | null;
}

const readCookie = (header: string | null | undefined, name: string): string | null => {
  if (!header) return null;
  for (const part of header.split(';')) {
    const [k, ...rest] = part.trim().split('=');
    if (k === name) return decodeURIComponent(rest.join('='));
  }
  return null;
};

export const jwtAuthMiddleware = (jwt: JwtService): MiddlewareHandler => async (c, next) => {
  const authHeader = c.req.header('authorization') ?? '';
  const m = /^Bearer\s+(.+)$/i.exec(authHeader);
  if (!m || !m[1]) {
    return c.json({ error: { code: 'auth_error', message: 'missing bearer token' } }, 401);
  }
  try {
    const claims = await jwt.verify(m[1]);
    const cookieHeader = c.req.header('cookie');
    const passphrase = readCookie(cookieHeader, 'raghub_passphrase');
    c.set('claims', claims);
    c.set('passphrase', passphrase);
    return await next();
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

export const getPassphrase = (c: Context): string | null => {
  const value = c.get('passphrase');
  return typeof value === 'string' && value.length > 0 ? value : null;
};