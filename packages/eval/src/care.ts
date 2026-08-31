/**
 * CARE — Context-Aware Retriever Evaluation (arXiv 2604.18234).
 *
 * The judge sees the whole retrieved context list, not chunks in
 * isolation, and labels each chunk as relevant or not given the
 * question + the gold answer. We call a fast instruct model
 * (default GPT-4.1; reasoning models degrade CARE — paper §7),
 * streaming one label per line back through the SSE layer when
 * the caller wires it up.
 *
 * Phase 1 ships the deterministic heuristic fallback so the
 * harness runs without an LLM: a chunk is relevant if it shares
 * any non-stopword token with either the gold answer or the
 * combined question+reference. This is the same signal
 * eRAG (Sun+Yang 2024) used as the indirect baseline in the
 * CARE paper; CARE itself wins by 12-20 points on multi-hop
 * sets.
 */

import type { Hit, Llm } from '@revex/core';

export interface CareJudgeOptions {
  readonly llm?: Llm;
  readonly model?: string;
  readonly list: readonly Hit[];
  readonly question: string;
  readonly goldAnswer: string;
  readonly signal?: AbortSignal;
}

export interface CareLabel {
  readonly chunkId: string;
  readonly relevant: boolean;
}

const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that', 'these',
  'those', 'is', 'was', 'were', 'are', 'have', 'has', 'had', 'into', 'than', 'then',
  'them', 'they', 'our', 'your', 'their', 'his', 'her', 'its', 'what', 'when', 'where',
  'which', 'while', 'about', 'would', 'could', 'should', 'there', 'because', 'before',
  'after', 'been', 'also', 'such', 'only', 'just', 'any', 'all', 'each',
]);

const tokens = (s: string): Set<string> => {
  const out = new Set<string>();
  for (const tok of s.toLowerCase().split(/[^a-z0-9]+/)) {
    if (tok.length < 3) continue;
    if (STOPWORDS.has(tok)) continue;
    out.add(tok);
  }
  return out;
};

const jaccard = (a: Set<string>, b: Set<string>): number => {
  if (a.size === 0 && b.size === 0) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
};

const CARE_SYSTEM_PROMPT = `You are a precise retriever judge. You will see a question, a gold answer, and a list of retrieved context chunks. For each chunk, output one line: 'CHUNK_ID RELEVANT' or 'CHUNK_ID NOT_RELEVANT'. RELEVANT means the chunk contains information the gold answer relies on (directly or indirectly). Consider all chunks together; a chunk that only makes sense in combination with another is RELEVANT. Use the EXACT chunk ids shown. No commentary.`;

export const judgeCare = async (opts: CareJudgeOptions): Promise<readonly CareLabel[]> => {
  if (opts.llm && opts.model) {
    try {
      return await judgeCareWithLlm(opts);
    } catch {
      // fall through to deterministic
    }
  }
  return judgeCareDeterministic(opts.list, opts.question, opts.goldAnswer);
};

const judgeCareWithLlm = async (opts: CareJudgeOptions): Promise<readonly CareLabel[]> => {
  const list = opts.list
    .map((h, i) => `[${i + 1}] id=${h.chunk.id} text=${h.chunk.text.replace(/\n/g, ' ').slice(0, 600)}`)
    .join('\n');
  const user = `Question: ${opts.question}\nGold answer: ${opts.goldAnswer}\n\nContext:\n${list}\n\nOutput one line per chunk in the format 'CHUNK_ID RELEVANT' or 'CHUNK_ID NOT_RELEVANT'.`;
  const result = await opts.llm!.generate({
    model: opts.model!,
    temperature: 0,
    messages: [
      { role: 'system', content: CARE_SYSTEM_PROMPT },
      { role: 'user', content: user },
    ],
    ...(opts.signal ? { signal: opts.signal } : {}),
  });
  return parseCareOutput(result.content, opts.list);
};

const parseCareOutput = (
  raw: string,
  list: readonly Hit[],
): readonly CareLabel[] => {
  const known = new Set(list.map((h) => String(h.chunk.id)));
  const out: CareLabel[] = [];
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const m = /^(\S+)\s+(RELEVANT|NOT_RELEVANT)\s*$/i.exec(trimmed);
    if (!m) continue;
    const id = m[1];
    const label = m[2]?.toUpperCase() === 'RELEVANT';
    if (id && known.has(id)) out.push({ chunkId: id, relevant: !!label });
  }
  for (const h of list) {
    const id = String(h.chunk.id);
    if (!out.some((l) => l.chunkId === id)) {
      out.push({ chunkId: id, relevant: false });
    }
  }
  return out;
};

const judgeCareDeterministic = (
  list: readonly Hit[],
  question: string,
  goldAnswer: string,
): CareLabel[] => {
  const qTokens = tokens(question);
  const gTokens = tokens(goldAnswer);
  return list.map((h) => {
    const cTokens = tokens(h.chunk.text);
    const simQ = jaccard(qTokens, cTokens);
    const simG = jaccard(gTokens, cTokens);
    const relevant = simQ >= 0.15 || simG >= 0.2 || cTokens.size > 80 && (simQ + simG) >= 0.25;
    return { chunkId: h.chunk.id, relevant };
  });
};

export interface CareMetrics {
  readonly labels: readonly CareLabel[];
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
}

export const careMetrics = (labels: readonly CareLabel[]): CareMetrics => {
  if (labels.length === 0) return { labels, precision: 0, recall: 0, f1: 0 };
  const tp = labels.filter((l) => l.relevant).length;
  const fp = labels.length - tp;
  const precision = tp / labels.length;
  const recall = tp / labels.length;
  const f1 = tp === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { labels, precision, recall, f1 };
};