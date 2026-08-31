/**
 * Security headers + CORS.
 *
 * Adds `X-Content-Type-Options`, `Referrer-Policy`,
 * `X-Frame-Options`, and a permissive CORS rule for the Next.js
 * dev origin. Production deployments behind a reverse proxy should
 * add CSP / HSTS at the proxy layer instead of the app.
 */

import type { Context, MiddlewareHandler } from 'hono';

export interface SecurityHeadersOptions {
  readonly allowOrigins?: readonly string[];
  readonly allowMethods?: readonly string[];
}

const DEFAULT_ORIGINS = [
  'http://localhost:3001',
  'http://127.0.0.1:3001',
];

export const securityHeadersMiddleware = (
  opts: SecurityHeadersOptions = {},
): MiddlewareHandler => {
  const origins = opts.allowOrigins ?? DEFAULT_ORIGINS;
  const methods = opts.allowMethods ?? ['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'];

  const isAllowed = (origin: string): boolean =>
    origins.includes(origin) || origins.includes('*');

  return async (c: Context, next) => {
    const origin = c.req.header('origin') ?? '';
    if (origin && isAllowed(origin)) {
      c.header('Access-Control-Allow-Origin', origin);
      c.header('Vary', 'Origin');
      c.header('Access-Control-Allow-Credentials', 'true');
      c.header('Access-Control-Allow-Methods', methods.join(', '));
      c.header('Access-Control-Allow-Headers', 'content-type, authorization, x-revex-path');
    }
    c.header('X-Content-Type-Options', 'nosniff');
    c.header('Referrer-Policy', 'no-referrer');
    c.header('X-Frame-Options', 'DENY');
    if (c.req.method === 'OPTIONS') {
      return c.body(null, 204);
    }
    return await next();
  };
};