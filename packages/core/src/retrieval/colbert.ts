/**
 * ColBERT-style late-interaction scoring.
 *
 * Classic ColBERT encodes every token with a small per-token
 * vector and scores query-document similarity as the max-similarity
 * over token pairs. The "late interaction" name means the per-token
 * similarity happens at query time, not at index time, which
 * trades compute for index-size.
 *
 * This module implements a simplified version: query and document
 * embeddings are token-level vectors. We compute MaxSim across
 * tokens and sum over query tokens, then normalise by query length.
 *
 * `lateInteractionScore(queryVecs, docVecs)` returns the score.
 * Higher is better. `lateInteractionRerank(query, hits, embedder)`
 * ranks `hits` by score and returns the top-k.
 */

import type { Hit } from '../domain/chunk.js';
import type { Embedder } from '../embedder/types.js';

const dot = (a: readonly number[], b: readonly number[]): number => {
  let s = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
};

export const lateInteractionScore = (
  queryVecs: readonly (readonly number[])[],
  docVecs: readonly (readonly number[])[],
): number => {
  if (queryVecs.length === 0 || docVecs.length === 0) return 0;
  let total = 0;
  for (const q of queryVecs) {
    let best = -Infinity;
    for (const d of docVecs) {
      const s = dot(q, d);
      if (s > best) best = s;
    }
    if (best !== -Infinity) total += best;
  }
  return total / queryVecs.length;
};

export const lateInteractionRerank = async (
  query: string,
  hits: readonly Hit[],
  embedder: Embedder & { embedTokens(text: string): Promise<readonly (readonly number[])[]> },
  topK: number = 10,
): Promise<readonly Hit[]> => {
  if (hits.length === 0) return [];
  const queryVecs = await embedder.embedTokens(query);
  const scored: { hit: Hit; score: number }[] = [];
  for (const h of hits) {
    const docVecs = await embedder.embedTokens(h.chunk.text);
    scored.push({ hit: h, score: lateInteractionScore(queryVecs, docVecs) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map((s) => ({ chunk: s.hit.chunk, score: s.score }));
};