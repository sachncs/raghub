/**
 * WorkspacePool — caches unlocked WorkspaceWithSettings handles per
 * (workspaceId, userId) pair for the lifetime of the daemon
 * process.
 *
 * Why this exists:
 *   - The server is stateless across requests, but every request
 *     needs the LLM settings that live inside the encrypted
 *     workspace_settings table.
 *   - Re-decrypting with scrypt(N=2¹⁵) on every request is
 *     measurable on hot paths.
 *   - The JWT alone can't decrypt the workspace — the server has to
 *     also know the passphrase. We solve this by stashing the
 *     passphrase in the cookie (P-07) and caching the unlocked
 *     handle keyed by user identity.
 *
 * The pool is bounded by `maxHandles`; oldest entries are evicted
 * on overflow. Closing a handle flushes its underlying connection.
 */

import {
  openEncryptedWorkspace,
  type WorkspaceId,
  type WorkspaceRegistry,
  type WorkspaceWithSettings,
} from '@revex/core';

export interface WorkspacePoolOptions {
  readonly registry: WorkspaceRegistry;
  readonly maxHandles?: number;
  readonly lockTimeoutMs?: number;
}

interface CacheKey {
  readonly workspaceId: string;
  readonly userId: string;
}

interface CacheEntry {
  readonly handle: WorkspaceWithSettings;
  readonly lastUsed: number;
}

const DEFAULT_MAX_HANDLES = 64;
const DEFAULT_LOCK_TIMEOUT_MS = 30_000;

export class WorkspacePool {
  private readonly registry: WorkspaceRegistry;
  private readonly maxHandles: number;
  private readonly lockTimeoutMs: number;
  private readonly entries = new Map<string, CacheEntry>();
  private readonly locks = new Map<string, Promise<WorkspaceWithSettings>>();

  constructor(opts: WorkspacePoolOptions) {
    this.registry = opts.registry;
    this.maxHandles = opts.maxHandles ?? DEFAULT_MAX_HANDLES;
    this.lockTimeoutMs = opts.lockTimeoutMs ?? DEFAULT_LOCK_TIMEOUT_MS;
  }

  /**
   * Get (or open) the unlocked workspace handle for the active
   * `(workspaceId, userId)`. Concurrent requests for the same key
   * share a single in-flight open promise.
   */
  public async get(input: {
    workspaceId: WorkspaceId;
    userId: string;
    passphrase: string;
  }): Promise<WorkspaceWithSettings> {
    const key = `${input.workspaceId}::${input.userId}`;
    if (process.env['REVEX_DEBUG_POOL']) {
      // eslint-disable-next-line no-console
      console.log(`[pool] get key=${key} cached=${this.entries.has(key)}`);
    }
    const cached = this.entries.get(key);
    if (cached) {
      cached.handle.close.length;
      this.entries.set(key, { handle: cached.handle, lastUsed: Date.now() });
      return cached.handle;
    }
    const inflight = this.locks.get(key);
    if (inflight) return inflight;

    const promise = this.open(input);
    this.locks.set(key, promise);
    try {
      const handle = await promise;
      this.evictIfNeeded();
      this.entries.set(key, { handle, lastUsed: Date.now() });
      return handle;
    } finally {
      this.locks.delete(key);
    }
  }

  public close(workspaceId: WorkspaceId, userId: string): void {
    const key = `${workspaceId}::${userId}`;
    const entry = this.entries.get(key);
    if (entry) {
      void entry.handle.close();
      this.entries.delete(key);
    }
  }

  public closeAll(): void {
    for (const entry of this.entries.values()) void entry.handle.close();
    this.entries.clear();
  }

  public size(): number {
    return this.entries.size;
  }

  private async open(input: {
    workspaceId: WorkspaceId;
    userId: string;
    passphrase: string;
  }): Promise<WorkspaceWithSettings> {
    const entry = await this.registry.resolve(input.workspaceId);
    if (!entry) {
      throw new Error(`workspace ${input.workspaceId} not registered`);
    }
    return openEncryptedWorkspace({
      path: entry.path,
      passphrase: input.passphrase,
    });
  }

  private evictIfNeeded(): void {
    if (this.entries.size < this.maxHandles) return;
    let oldestKey: string | null = null;
    let oldestTs = Number.POSITIVE_INFINITY;
    for (const [k, v] of this.entries.entries()) {
      if (v.lastUsed < oldestTs) {
        oldestTs = v.lastUsed;
        oldestKey = k;
      }
    }
    if (oldestKey !== null) {
      const entry = this.entries.get(oldestKey);
      if (entry) void entry.handle.close();
      this.entries.delete(oldestKey);
    }
  }
}

export const _lockTimeout = (pool: WorkspacePool): number => pool['lockTimeoutMs'];
void _lockTimeout;
void ({} as CacheKey);