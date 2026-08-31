/**
 * Memory / admin routes — `/v1/admin/stats` and `/v1/admin/vacuum`.
 *
 * `stats` is the surface the /memory dashboard reads. It returns
 * the document + chunk counts, + total embedding bytes on disk,
 * per-status counts, and the last-ingested timestamp so the UI
 * can render the "Memory" panel.
 *
 * `vacuum` is a manual escape hatch: it reclaims SQLite pages
 * after a large delete (documents or chunks). Bounded by the
 * admin role on the workspace.
 */

import { Hono } from 'hono';

import type { StoreStats, VectorStore } from '@revex/core';

import { getClaims } from '../middleware/auth.js';
import { workspaceContextFrom } from '../workspace-context.js';
import type { WorkspacePool } from '../workspace-pool.js';

export interface MemoryRouteDeps {
  readonly pool: WorkspacePool;
  readonly vectorStore?: VectorStore | null;
}

export interface MemoryRouteDeps {
  readonly pool: WorkspacePool;
  readonly vectorStore?: VectorStore | null;
}

export interface MemoryStatsResponse {
  readonly workspaceId: string;
  readonly documentCount: number;
  readonly chunkCount: number;
  readonly totalTokens: number;
  readonly embeddingBytes: number;
  readonly bytesOnDisk: number;
  readonly lastIngestedAt: number | null;
  readonly statusCounts: Readonly<Record<string, number>>;
  readonly sources: ReadonlyArray<{
    readonly id: string;
    readonly name: string;
    readonly kind: string;
    readonly chunkCount: number;
  }>;
  readonly capacity: {
    readonly perWorkspaceDocumentsSoftLimit: number;
    readonly perWorkspaceChunksSoftLimit: number;
    readonly note: string;
  };
}

const SOFT_LIMITS = {
  documents: 50_000,
  chunks: 5_000_000,
};

export const memoryRoutes = (deps: MemoryRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/admin/stats', async (c) => {
    const claims = getClaims(c);
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: null as never,
      vectorStore: deps.vectorStore ?? null,
    });
    if (!ctx.vectorStore) {
      return c.json({ error: { code: 'revex_error', message: 'vector store not available' } }, 503);
    }
    const stats: StoreStats = await ctx.vectorStore.stats(ctx.workspaceId);

    const docs = await ctx.documentStore.listForUser(ctx.workspaceId, ctx.userId);
    const sourceMap = new Map<string, { id: string; name: string; kind: string; chunkCount: number }>();
    for (const d of docs) {
      const key = `${d.filename}:${d.mimeType}`;
      const existing = sourceMap.get(key);
      if (existing) {
        existing.chunkCount += 1;
      } else {
        sourceMap.set(key, {
          id: d.id,
          name: d.filename,
          kind: d.mimeType,
          chunkCount: 1,
        });
      }
    }

    const response: MemoryStatsResponse = {
      workspaceId: String(claims.workspace_id),
      documentCount: stats.documentCount,
      chunkCount: stats.chunkCount,
      totalTokens: stats.totalTokenEstimate,
      embeddingBytes: stats.embeddingBytes,
      bytesOnDisk: stats.bytesOnDisk,
      lastIngestedAt: stats.lastIngestedAt,
      statusCounts: stats.statusCounts,
      sources: Array.from(sourceMap.values()).slice(0, 20),
      capacity: {
        perWorkspaceDocumentsSoftLimit: SOFT_LIMITS.documents,
        perWorkspaceChunksSoftLimit: SOFT_LIMITS.chunks,
        note: 'Soft limits are advisory. Production deployments may set stricter quotas via REVEX_QUOTAS_DOC / REVEX_QUOTAS_CHUNK.',
      },
    };
    return c.json(response);
  });

  app.post('/v1/admin/vacuum', async (c) => {
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: null as never,
      vectorStore: deps.vectorStore ?? null,
    });
    const handle = ctx.handle;
    type DbWithVacuum = { pragma: (sql: string) => unknown; exec?: (sql: string) => unknown };
    const db = handle.db as unknown as DbWithVacuum;
    const before = db.pragma('integrity_check');
    db.exec?.('VACUUM;');
    db.exec?.('ANALYZE;');
    const after = db.pragma('integrity_check');
    return c.json({
      workspaceId: String(ctx.workspaceId),
      integrityBefore: before,
      integrityAfter: after,
      vacuumedAt: Date.now(),
    });
  });

  return app;
};