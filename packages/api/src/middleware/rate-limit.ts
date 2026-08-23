/**
 * Rate limiting middleware.
 *
 * Simple sliding-window counter keyed by IP + workspace. Default
 * limits: 60 req/min/IP and 600 req/min/workspace. Override via
 * `RateLimitOptions`.
 *
 * Memory-only — restart the daemon to reset counters. Production
 * deployments should swap this for a Redis-backed bucket; this
 * implementation is the single-process baseline.
 */

import type { Context, MiddlewareHandler } from 'hono';

interface Bucket {
  readonly key: string;
  count: number;
  resetAt: number;
}

export interface RateLimitOptions {
  readonly perIpPerMinute?: number;
  readonly perWorkspacePerMinute?: number;
  readonly bypassPaths?: readonly string[];
}

const DEFAULTS: Required<Omit<RateLimitOptions, 'bypassPaths'>> = {
  perIpPerMinute: 60,
  perWorkspacePerMinute: 600,
};

const WINDOW_MS = 60_000;

const clientIp = (c: Context): string => {
  const xff = c.req.header('x-forwarded-for');
  if (xff) return xff.split(',')[0]?.trim() ?? 'unknown';
  return 'unknown';
};

export const rateLimitMiddleware = (opts: RateLimitOptions = {}): MiddlewareHandler => {
  const perIpLimit = opts.perIpPerMinute ?? DEFAULTS.perIpPerMinute;
  const perWsLimit = opts.perWorkspacePerMinute ?? DEFAULTS.perWorkspacePerMinute;
  const ipBuckets = new Map<string, Bucket>();
  const wsBuckets = new Map<string, Bucket>();
  const bypass = new Set(opts.bypassPaths ?? ['/health', '/readyz']);

  const take = (buckets: Map<string, Bucket>, key: string, limit: number): { allowed: boolean; resetAt: number } => {
    const now = Date.now();
    let bucket = buckets.get(key);
    if (!bucket || bucket.resetAt < now) {
      bucket = { key, count: 0, resetAt: now + WINDOW_MS };
      buckets.set(key, bucket);
    }
    bucket.count++;
    return { allowed: bucket.count <= limit, resetAt: bucket.resetAt };
  };

  const cleanup = (buckets: Map<string, Bucket>): void => {
    const now = Date.now();
    for (const [k, b] of buckets.entries()) {
      if (b.resetAt < now) buckets.delete(k);
    }
  };

  setInterval(() => {
    cleanup(ipBuckets);
    cleanup(wsBuckets);
  }, WINDOW_MS).unref();

  return async (c, next) => {
    const path = c.req.path;
    if (bypass.has(path)) return await next();

    const ip = clientIp(c);
    const ipCheck = take(ipBuckets, ip, perIpLimit);
    if (!ipCheck.allowed) {
      c.header('retry-after', String(Math.ceil((ipCheck.resetAt - Date.now()) / 1000)));
      return c.json({ error: { code: 'rate_limit', message: 'per-IP limit exceeded' } }, 429);
    }

    const auth = c.req.header('authorization') ?? '';
    const wsMatch = /^[Bb]earer\s+(.+)$/.exec(auth);
    if (wsMatch) {
      try {
        const payload = JSON.parse(Buffer.from((wsMatch[1] ?? '').split('.')[1] ?? '', 'base64').toString('utf8')) as { workspace_id?: string };
        if (payload.workspace_id) {
          const wsCheck = take(wsBuckets, payload.workspace_id, perWsLimit);
          if (!wsCheck.allowed) {
            c.header('retry-after', String(Math.ceil((wsCheck.resetAt - Date.now()) / 1000)));
            return c.json({ error: { code: 'rate_limit', message: 'per-workspace limit exceeded' } }, 429);
          }
        }
      } catch {
        // Ignore — JWT verification happens in the auth middleware.
      }
    }

    return await next();
  };
};