/**
 * In-memory query cache.
 *
 * Caches the (question, retrieval-options, user-id) tuple to
 * the resulting `Hit[]`. Bypassed when the caller marks
 * `noCache: true`. Replace with Redis for multi-process
 * deployments.
 */

import type { Hit } from '@revex/core';

export interface CacheKey {
  readonly question: string;
  readonly workspaceId: string;
  readonly userId: string;
  readonly topK: number;
}

const hashKey = (k: CacheKey): string =>
  `${k.workspaceId}::${k.userId}::${k.topK}::${k.question}`;

interface Entry {
  readonly hits: readonly Hit[];
  readonly expiresAt: number;
}

export class QueryCache {
  private readonly store = new Map<string, Entry>();

  constructor(private readonly ttlMs: number = 60_000) {}

  get(key: CacheKey): readonly Hit[] | null {
    const entry = this.store.get(hashKey(key));
    if (!entry) return null;
    if (entry.expiresAt < Date.now()) {
      this.store.delete(hashKey(key));
      return null;
    }
    return entry.hits;
  }

  set(key: CacheKey, hits: readonly Hit[]): void {
    this.store.set(hashKey(key), { hits, expiresAt: Date.now() + this.ttlMs });
  }

  invalidate(workspaceId?: string): void {
    if (!workspaceId) {
      this.store.clear();
      return;
    }
    for (const k of Array.from(this.store.keys())) {
      if (k.startsWith(`${workspaceId}::`)) this.store.delete(k);
    }
  }
}