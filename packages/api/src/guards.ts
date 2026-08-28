/**
 * requireStore — guards a route handler when its backing store is
 * null. Routes that haven't been migrated to per-request
 * WorkspaceContext resolution still take flat deps at boot time;
 * if the server boots before any workspace is registered, those
 * flat deps are null and the route would crash. This helper
 * converts that into a clean 503.
 */

import type { Context } from 'hono';

export class StoreUnavailableError extends Error {
  constructor(public readonly label: string) {
    super(`store unavailable: ${label}`);
    this.name = 'StoreUnavailableError';
  }
}

export const requireStore = <T>(label: string, value: T | null | undefined): T => {
  if (value === null || value === undefined) {
    throw new StoreUnavailableError(label);
  }
  return value;
};

export const isStoreUnavailable = (e: unknown): boolean =>
  e instanceof StoreUnavailableError;

export const storeUnavailableResponse = (c: Context, label: string): Response =>
  c.json(
    { error: { code: 'configuration_error', message: `workspace not bootstrapped: ${label}` } },
    503,
  );