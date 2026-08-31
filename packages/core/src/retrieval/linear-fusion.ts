/**
 * Linear weighted-sum fusion.
 *
 * Linear fusion assigns each hit a score = α * denseScore + β *
 * sparseScore, normalised by the per-source max. RRF is usually
 * a better default (no normalisation needed), but linear fusion
 * is cheaper and useful for benchmarks that expect raw scores.
 */

import type { Hit } from '../domain/chunk.js';

export interface LinearFusionOptions {
  readonly denseWeight: number;
  readonly sparseWeight: number;
}

export const DEFAULT_LINEAR: LinearFusionOptions = {
  denseWeight: 0.6,
  sparseWeight: 0.4,
};

const KEY = Symbol('denseRank');
const KEY2 = Symbol('sparseRank');

export const linearFusion = (
  dense: readonly Hit[],
  sparse: readonly Hit[],
  opts: LinearFusionOptions = DEFAULT_LINEAR,
): readonly Hit[] => {
  const denseMax = dense.reduce((m, h) => Math.max(m, h.score), 0) || 1;
  const sparseMax = sparse.reduce((m, h) => Math.max(m, h.score), 0) || 1;
  const merged = new Map<string, Hit & { [KEY]?: number; [KEY2]?: number }>();
  dense.forEach((h, i) => {
    const existing = merged.get(h.chunk.id) ?? h;
    merged.set(h.chunk.id, { ...existing, score: h.score / denseMax, [KEY]: i + 1 });
  });
  sparse.forEach((h, i) => {
    const existing = merged.get(h.chunk.id) ?? h;
    merged.set(h.chunk.id, { ...existing, score: h.score / sparseMax, [KEY2]: i + 1 });
  });
  return Array.from(merged.values())
    .map((h) => ({
      chunk: h.chunk,
      score:
        h[KEY] !== undefined
          ? opts.denseWeight * h.score + opts.sparseWeight * ((h[KEY2] ?? 0) > 0 ? 1 / h[KEY2]! : 0)
          : opts.sparseWeight * h.score,
    }))
    .sort((a, b) => b.score - a.score);
};