/**
 * Per-request tenant context.
 *
 * Stored in `AsyncLocalStorage` so deeply-nested store calls pick it
 * up without threading the value through every signature. The
 * legacy Python implementation uses `contextvars.ContextVar`; this
 * is the Node equivalent.
 *
 * Active isolation strategy: RowLevel only. There is no other.
 */

import { AsyncLocalStorage } from 'node:async_hooks';

import { AuthorizationError } from '../errors/index.js';
import type { TenantId, UserId } from '../domain/ids.js';

export interface TenantContextValue {
  readonly tenantId: TenantId;
  readonly userId: UserId | null;
  readonly isAdmin: boolean;
  readonly sessionId: string | null;
}

const storage = new AsyncLocalStorage<TenantContextValue>();

export const runWithTenant = <T>(
  ctx: TenantContextValue,
  fn: () => T | Promise<T>,
): T | Promise<T> => storage.run(ctx, fn);

export const runWithTenantAsync = async <T>(
  ctx: TenantContextValue,
  fn: () => Promise<T>,
): Promise<T> => storage.run(ctx, fn);

export const currentTenant = (): TenantContextValue | undefined => storage.getStore();

export const requireTenant = (): TenantContextValue => {
  const ctx = storage.getStore();
  if (!ctx) throw new AuthorizationError('no tenant context bound');
  return ctx;
};

/**
 * Build a tenant context from explicit fields.
 *
 * Use at the request boundary — typically the API middleware that
 * decoded the JWT.
 */
export const tenantContext = (input: {
  tenantId: TenantId;
  userId: UserId | null;
  isAdmin: boolean;
  sessionId?: string | null;
}): TenantContextValue => ({
  tenantId: input.tenantId,
  userId: input.userId,
  isAdmin: input.isAdmin,
  sessionId: input.sessionId ?? null,
});