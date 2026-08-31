/**
 * Finance + Frames benchmark harnesses.
 *
 * Both expose the same contract: a `run(samples, llm, embedder,
 * retrieveFn)` that takes a list of QA pairs and an answer-
 * generator, runs each through retrieval + generation, and
 * returns per-sample metrics + aggregate scores.
 *
 * The samples themselves ship as JSONL; the harness reads them
 * from disk or accepts an in-memory list. Phase 1 ships the
 * harness + the loader; populating the datasets is a follow-up
 * commit that points at public eval-set mirrors.
 */

import { type Hit, type Llm, type Retrieval, User, UserRole } from '@revex/core';

import { computeMetrics, type Metrics } from './metrics.js';

export interface QASample {
  readonly id: string;
  readonly question: string;
  readonly goldAnswer: string;
  readonly goldIds: readonly string[];
  readonly metadata?: Readonly<Record<string, string>>;
}

export interface SampleResult {
  readonly sample: QASample;
  readonly metrics: Metrics;
  readonly latencyMs: number;
  readonly answer: string;
}

export interface AggregateMetrics {
  readonly count: number;
  readonly recallAtK: number;
  readonly precisionAtK: number;
  readonly mrr: number;
  readonly faithfulness: number;
  readonly answerCorrectness: number;
}

export const aggregate = (results: readonly SampleResult[]): AggregateMetrics => {
  if (results.length === 0) {
    return {
      count: 0,
      recallAtK: 0,
      precisionAtK: 0,
      mrr: 0,
      faithfulness: 0,
      answerCorrectness: 0,
    };
  }
  const sum = (m: keyof Metrics): number =>
    results.reduce((acc, r) => acc + r.metrics[m], 0) / results.length;
  return {
    count: results.length,
    recallAtK: sum('recallAtK'),
    precisionAtK: sum('precisionAtK'),
    mrr: sum('mrr'),
    faithfulness: sum('faithfulness'),
    answerCorrectness: sum('answerCorrectness'),
  };
};

export interface RunOptions {
  readonly retrieval: Retrieval;
  readonly llm: Llm;
  readonly model: string;
  readonly k?: number;
}

const extractClaims = (text: string): string[] =>
  text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

const makeSystemUser = (): import('@revex/core').User =>
  new User({
    id: '__system__' as never,
    workspaceId: '__system__' as never,
    email: '',
    role: UserRole.Admin,
    allowedCompanies: [],
    createdAt: new Date(),
  });

export const runSamples = async (
  samples: readonly QASample[],
  opts: RunOptions,
): Promise<readonly SampleResult[]> => {
  const results: SampleResult[] = [];
  for (const sample of samples) {
    const start = Date.now();
    const systemUser = makeSystemUser();
    const hits: Hit[] = await opts.retrieval.retrieve(systemUser, sample.question, opts.k ?? 10);
    const ctx = hits.map((h) => h.chunk.text).join('\n');
    const answerResult = await opts.llm.generate({
      model: opts.model,
      temperature: 0,
      messages: [
        { role: 'system', content: 'Answer concisely using the provided context.' },
        { role: 'user', content: `Context:\n${ctx}\n\nQuestion: ${sample.question}` },
      ],
    });
    const answer = answerResult.content;
    const metrics = computeMetrics({
      hits: hits.map((h: Hit) => ({ id: h.chunk.id })),
      groundTruthIds: sample.goldIds,
      answer,
      reference: sample.goldAnswer,
      contextClaims: extractClaims(answer),
      contextText: ctx,
    });
    results.push({ sample, metrics, latencyMs: Date.now() - start, answer });
  }
  return results;
};

export const loadJsonl = (raw: string): readonly QASample[] => {
  const out: QASample[] = [];
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const obj = JSON.parse(trimmed) as Record<string, unknown>;
      out.push({
        id: String(obj['id'] ?? ''),
        question: String(obj['question'] ?? ''),
        goldAnswer: String(obj['gold_answer'] ?? obj['answer'] ?? ''),
        goldIds: Array.isArray(obj['gold_ids']) ? obj['gold_ids'].map(String) : [],
        ...(obj['metadata'] && typeof obj['metadata'] === 'object'
          ? { metadata: obj['metadata'] as Record<string, string> }
          : {}),
      });
    } catch {
      // skip malformed lines
    }
  }
  return out;
};

export const loadJsonlFile = async (path: string): Promise<readonly QASample[]> => {
  const { promises: fs } = await import('node:fs');
  const raw = await fs.readFile(path, 'utf8');
  return loadJsonl(raw);
};

export interface FinanceRunResult {
  readonly benchmark: 'financebench';
  readonly aggregate: AggregateMetrics;
  readonly results: readonly SampleResult[];
}

export const runFinance = async (
  samples: readonly QASample[],
  opts: RunOptions,
): Promise<FinanceRunResult> => {
  const results = await runSamples(samples, opts);
  return { benchmark: 'financebench', aggregate: aggregate(results), results };
};

export interface FramesRunResult {
  readonly benchmark: 'frames';
  readonly aggregate: AggregateMetrics;
  readonly results: readonly SampleResult[];
}

export const runFrames = async (
  samples: readonly QASample[],
  opts: RunOptions,
): Promise<FramesRunResult> => {
  const results = await runSamples(samples, opts);
  return { benchmark: 'frames', aggregate: aggregate(results), results };
};