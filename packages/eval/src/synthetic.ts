/**
 * Synthetic QA dataset generator.
 *
 * Given a list of source passages and a question generator (the
 * LLM), produces `count` QASamples. Each sample's `goldIds` is
 * the index of the passage that was its source.
 */

import type { Llm } from '@revex/core';

export interface SyntheticSample {
  readonly id: string;
  readonly question: string;
  readonly goldAnswer: string;
  readonly goldIds: readonly string[];
  readonly sourceIndex: number;
}

export interface SyntheticOptions {
  readonly llm: Llm;
  readonly model?: string;
  readonly count: number;
}

const buildPrompt = (passage: string): string => {
  return (
    'Generate one factual question and its gold answer based on the following passage. ' +
    'Reply with JSON only — no prose. Schema: {"question": "<placeholder>", "answer": "<placeholder>"}' +
    '\n\nPassage:\n' + passage
  );
};

const id = (): string => `syn_${Math.random().toString(36).slice(2, 10)}`;

export async function generateSynthetic(
  passages: readonly string[],
  opts: SyntheticOptions,
): Promise<readonly SyntheticSample[]> {
  if (passages.length === 0) return [];
  const out: SyntheticSample[] = [];
  for (let i = 0; i < opts.count; i++) {
    const idx = i % passages.length;
    const passage = passages[idx] ?? '';
    if (!passage) continue;
    const r = await opts.llm.generate({
      model: opts.model ?? opts.llm.model,
      temperature: 0.4,
      messages: [{ role: 'user', content: buildPrompt(passage) }],
      responseFormat: { type: 'json_object' },
    });
    try {
      const parsed = JSON.parse(r.content) as { question?: string; answer?: string };
      if (parsed.question && parsed.answer) {
        out.push({
          id: id(),
          question: parsed.question,
          goldAnswer: parsed.answer,
          goldIds: [`p${idx}`],
          sourceIndex: idx,
        });
      }
    } catch {
      /* skip */
    }
  }
  return out;
};