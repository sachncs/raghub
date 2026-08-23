/**
 * Lost-in-the-middle sensitivity probe (arXiv 2605.27105).
 *
 * For a fixed query, place the same gold-relevant chunk at
 * every position in a list of size k and measure answer
 * accuracy (or answer-correctness proxy) at each position. The
 * difference between the best and worst position is the
 * sensitivity; the paper's findings are that sensitivity is
 * high for older reasoning models and low for modern ones.
 *
 * Phase 1 implements the probe with an LLM-judge-friendly answer
 * proxy: distance-from-gold via token overlap. Production
 * callers swap in their preferred judge.
 */

import type { Hit, Llm } from '@raghub/core';

export interface LimOptions {
  readonly llm?: Llm;
  readonly model?: string;
  readonly goldChunkIds: readonly string[];
  readonly candidateChunks: readonly Hit[];
  readonly listSize: number;
  readonly query: string;
  readonly reference?: string;
}

export interface LimSample {
  readonly position: number;
  readonly accuracy: number;
}

export const lostInMiddleProbe = async (opts: LimOptions): Promise<readonly LimSample[]> => {
  if (opts.listSize <= 0 || opts.candidateChunks.length === 0) return [];
  const goldSet = new Set(opts.goldChunkIds);
  const samples: LimSample[] = [];
  for (let pos = 0; pos < opts.listSize; pos++) {
    const pool = [...opts.candidateChunks];
    const placed: Hit[] = [];
    for (let i = 0; i < opts.listSize; i++) {
      if (i === pos) {
        const gold = opts.candidateChunks.find((h) => goldSet.has(h.chunk.id));
        if (gold) placed.push(gold);
        else placed.push(pool.shift() ?? opts.candidateChunks[0]!);
      } else {
        const candidate = pool.shift();
        if (candidate) placed.push(candidate);
      }
    }
    const acc = await scoreList(opts, placed);
    samples.push({ position: pos, accuracy: acc });
  }
  return samples;
};

const scoreList = async (opts: LimOptions, list: readonly Hit[]): Promise<number> => {
  if (opts.reference) {
    const ctx = list.map((h) => h.chunk.text).join(' ');
    const overlap = tokens(opts.reference);
    let hit = 0;
    for (const t of overlap) if (ctx.toLowerCase().includes(t)) hit++;
    const base = opts.reference.split(/\s+/).length || 1;
    return hit / base;
  }
  if (opts.llm && opts.model) {
    try {
      const result = await opts.llm.generate({
        model: opts.model,
        temperature: 0,
        messages: [
          {
            role: 'system',
            content: 'You are a precise grader. Reply 1 or 0 only — 1 if the context contains the answer, 0 otherwise.',
          },
          {
            role: 'user',
            content: `Question: ${opts.query}\nContext: ${list.map((h) => h.chunk.text.slice(0, 200)).join(' | ')}`,
          },
        ],
      });
      return /\b1\b/.test(result.content) ? 1 : 0;
    } catch {
      return 0;
    }
  }
  return 0;
};

const tokens = (s: string): readonly string[] =>
  s
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 2);