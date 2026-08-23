/**
 * Retrieval pipeline — dense + sparse + RRF + RBAC.
 *
 * Mirrors the legacy `raghub/retrieval/pipeline.py` with the
 * post-redesign invariant: every read is filtered by a `StoreFilter`
 * built from the active user, and hybrid fusion happens via RRF on
 * the chunk-id axis.
 */

import type { Embedder } from '../embedder/index.js';
import type { Hit } from '../domain/index.js';
import type { User } from '../domain/index.js';
import { allowedCompanyFilter } from './rbac.js';
import { reciprocalRankFusion } from './rrf.js';
import type { VectorStore } from '../stores/index.js';

export interface RetrievalOptions {
  readonly topK: number;
  readonly denseWeight: number;
  readonly sparseWeight: number;
  readonly rrfK: number;
}

const DEFAULTS = {
  topK: 10,
  denseWeight: 0.6,
  sparseWeight: 0.4,
  rrfK: 60,
} as const;

export class Retrieval {
  private readonly embedder: Embedder;
  private readonly store: VectorStore;
  private readonly opts: RetrievalOptions;

  constructor(embedder: Embedder, store: VectorStore, opts?: Partial<RetrievalOptions>) {
    this.embedder = embedder;
    this.store = store;
    this.opts = { ...DEFAULTS, ...opts };
  }

  public async retrieve(user: User, question: string, topK?: number): Promise<Hit[]> {
    const k = topK ?? this.opts.topK;
    const filter = allowedCompanyFilter(user);
    const vec = await this.embedder.embedQuery(question);

    const dense = await this.store.searchVector({
      vector: vec,
      topK: k * 2,
      filter,
    });
    const keyword = await this.store.searchKeyword({
      query: question,
      topK: k * 2,
      filter,
    });

    const denseList = dense.map((h) => ({ id: h.chunk.id }));
    const keywordList = keyword.map((h) => ({ id: h.chunkId }));
    const fused = reciprocalRankFusion<Hit['chunk']['id']>([denseList, keywordList], this.opts.rrfK);

    const byId = new Map(dense.map((h) => [h.chunk.id, h]));
    const out: Hit[] = [];
    for (const id of fused) {
      const hit = byId.get(id);
      if (hit) {
        out.push(hit);
        if (out.length >= k) break;
      }
    }
    return out;
  }
}