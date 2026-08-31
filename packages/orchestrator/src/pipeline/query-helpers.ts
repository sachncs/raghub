/**
 * Query pipeline helpers.
 *
 * Shapes the retrieval result for the generator. Compresses
 * `Hit[]` into the `{text, sources}` pair the generator agent
 * expects, dedupes identical chunk ids, and preserves citation
 * numbers.
 */

import type { Hit } from '@revex/core';

export interface QueryContext {
  readonly hits: readonly Hit[];
  readonly topK?: number;
  readonly perHitChars?: number;
}

export interface QueryContextResult {
  readonly text: string;
  readonly sources: readonly { readonly citation: number; readonly chunkId: string; readonly documentId: string }[];
  readonly truncated: boolean;
}

export const shapeContext = (input: QueryContext): QueryContextResult => {
  const topK = input.topK ?? 5;
  const perHitChars = input.perHitChars ?? 1200;
  const seen = new Set<string>();
  const sources: { citation: number; chunkId: string; documentId: string }[] = [];
  const blocks: string[] = [];
  let truncated = false;
  let n = 0;
  for (const h of input.hits) {
    if (n >= topK) break;
    if (seen.has(h.chunk.id)) continue;
    seen.add(h.chunk.id);
    n += 1;
    let text = h.chunk.text;
    if (text.length > perHitChars) {
      text = text.slice(0, perHitChars);
      truncated = true;
    }
    sources.push({ citation: n, chunkId: h.chunk.id, documentId: h.chunk.documentId });
    blocks.push(`[${n}] ${text}`);
  }
  return {
    text: blocks.join('\n\n'),
    sources,
    truncated,
  };
};