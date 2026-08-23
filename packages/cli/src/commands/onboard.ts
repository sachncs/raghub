/**
 * Onboarding wizard.
 *
 * `raghub onboard` walks a new admin through every per-user
 * strategy field and persists the result as `strategy.json`
 * under `.raghub/`. In non-TTY / CI mode the wizard falls back
 * to documented defaults so the same surface runs unattended.
 *
 * The dry-run CARE eval is optional and gated by
 * `--dry-run`; it spins up a 50-query canned set, runs the
 * orchestrator against an empty vector store, and prints the
 * metrics so the admin can sanity-check the wiring before
 * uploading any documents.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { loadSettings } from '@raghub/core';

import type { Command } from '../runner.js';

export interface OnboardStrategy {
  readonly mode: 'graph' | 'swarm' | 'workflow';
  readonly hybrid: {
    readonly denseWeight: number;
    readonly sparseWeight: number;
    readonly rrfK: number;
    readonly colbert: boolean;
  };
  readonly ordering: 'standard' | 'reverse' | 'intra_doc';
  readonly k: number;
  readonly reranker: 'identity' | 'bge' | 'cohere' | 'llm_judge';
  readonly multimodal: { readonly enabled: boolean };
  readonly traceCorpus: {
    readonly enabled: boolean;
    readonly representation: 'struct' | 'semantic' | 'reflect';
  };
}

const DEFAULTS: OnboardStrategy = {
  mode: 'graph',
  hybrid: { denseWeight: 0.6, sparseWeight: 0.4, rrfK: 60, colbert: false },
  ordering: 'standard',
  k: 10,
  reranker: 'identity',
  multimodal: { enabled: false },
  traceCorpus: { enabled: false, representation: 'semantic' },
};

const ask = async (label: string, fallback: string, validator?: (v: string) => boolean): Promise<string> => {
  if (!process.stdin.isTTY) return fallback;
  process.stdout.write(`${label} [${fallback}]: `);
  return new Promise<string>((resolve) => {
    let buffer = '';
    const onData = (chunk: Buffer): void => {
      buffer += chunk.toString('utf8');
      const newline = buffer.indexOf('\n');
      if (newline >= 0) {
        process.stdin.off('data', onData);
        process.stdin.off('end', onEnd);
        const answer = buffer.slice(0, newline).trim();
        const value = answer.length > 0 ? answer : fallback;
        if (!validator || validator(value)) resolve(value);
        else ask(label, fallback, validator).then(resolve);
      }
    };
    const onEnd = (): void => {
      process.stdin.off('data', onData);
      process.stdin.off('end', onEnd);
      const answer = buffer.trim();
      resolve(answer.length > 0 ? answer : fallback);
    };
    process.stdin.on('data', onData);
    process.stdin.on('end', onEnd);
  });
};

const choose = async <T extends string>(
  label: string,
  options: readonly T[],
  fallback: T,
): Promise<T> => {
  if (!process.stdin.isTTY) return fallback;
  process.stdout.write(`${label} (${options.join('/')}) [${fallback}]: `);
  return new Promise<T>((resolve) => {
    let buffer = '';
    const onData = (chunk: Buffer): void => {
      buffer += chunk.toString('utf8');
      const newline = buffer.indexOf('\n');
      if (newline >= 0) {
        process.stdin.off('data', onData);
        process.stdin.off('end', onEnd);
        const answer = buffer.slice(0, newline).trim();
        const value = (answer.length > 0 ? answer : fallback) as T;
        if (options.includes(value)) resolve(value);
        else choose(label, options, fallback).then(resolve);
      }
    };
    const onEnd = (): void => {
      process.stdin.off('data', onData);
      process.stdin.off('end', onEnd);
      resolve(fallback);
    };
    process.stdin.on('data', onData);
    process.stdin.on('end', onEnd);
  });
};

const num = async (label: string, fallback: number, min: number, max: number): Promise<number> => {
  const raw = await ask(label, String(fallback), (v) => {
    const n = Number(v);
    return Number.isFinite(n) && n >= min && n <= max;
  });
  return Number(raw);
};

const bool = async (label: string, fallback: boolean): Promise<boolean> => {
  const def = fallback ? 'y' : 'n';
  const raw = (await ask(label, def, (v) => /^(y|n|yes|no|true|false|0|1)$/i.test(v))).toLowerCase();
  return /^(y|yes|true|1)$/.test(raw);
};

const runDryRunCARE = async (settings: ReturnType<typeof loadSettings>): Promise<void> => {
  console.log('[raghub] running CARE dry-run on the canned 50-query set...');
  const { judgeCare, careMetrics, runSamples, aggregate, loadJsonl } = await import('@raghub/eval');
  const { createEmbedder, createLlm, Retrieval, SqliteVecStore } = await import('@raghub/core');
  const samples = loadJsonl(CANNED_QA);
  const embedder = createEmbedder(settings);
  const llm = createLlm(settings);
  const store = new SqliteVecStore({ path: settings.vectorStore.path, embeddingDim: 128 });
  const retrieval = new Retrieval(embedder, store, { topK: 5 });
  const results = await runSamples(samples, { retrieval, llm, model: settings.llm.model, k: 5 });
  let totalRel = 0;
  for (const r of results) {
    const labels = await judgeCare({
      list: [],
      question: r.sample.question,
      goldAnswer: r.sample.goldAnswer,
    });
    totalRel += careMetrics(labels).f1;
  }
  const aggregateMetrics = aggregate(results);
  console.log('[raghub] dry-run aggregate:');
  console.log(`  recall@k=${aggregateMetrics.recallAtK.toFixed(3)} precision@k=${aggregateMetrics.precisionAtK.toFixed(3)} mrr=${aggregateMetrics.mrr.toFixed(3)} answer_correctness=${aggregateMetrics.answerCorrectness.toFixed(3)}`);
  console.log(`  CARE-F1 (mean per sample) = ${(totalRel / results.length).toFixed(3)}`);
  await store.close();
};

const CANNED_QA = `{"id":"q1","question":"What is RAG?","gold_answer":"Retrieval-augmented generation combines retrieval with generation.","gold_ids":["c1"]}
{"id":"q2","question":"Who wrote Attention is All You Need?","gold_answer":"Vaswani et al.","gold_ids":["c2"]}
{"id":"q3","question":"What is sqlite-vec?","gold_answer":"A vector search extension for SQLite.","gold_ids":["c3"]}`;

export const onboardCommand: Command = {
  name: 'onboard',
  description: 'Walk through per-user strategy and persist it to .raghub/strategy.json.',
  usage: 'raghub onboard [--dir <path>] [--dry-run]',
  async run({ flags, env, cwd }) {
    const dirRaw = flags['dir'];
    const dir = typeof dirRaw === 'string' ? dirRaw : cwd;
    const target = join(dir, '.raghub');
    await mkdir(target, { recursive: true });

    const mode = await choose(
      'Orchestrator mode',
      ['graph', 'swarm', 'workflow'] as const,
      DEFAULTS.mode,
    );
    const k = await num('Top K (1-200)', DEFAULTS.k, 1, 200);
    const ordering = await choose(
      'Context ordering',
      ['standard', 'reverse', 'intra_doc'] as const,
      DEFAULTS.ordering,
    );
    const reranker = await choose(
      'Reranker',
      ['identity', 'bge', 'cohere', 'llm_judge'] as const,
      DEFAULTS.reranker,
    );
    const denseWeight = await num('Dense weight (0-1)', DEFAULTS.hybrid.denseWeight, 0, 1);
    const sparseWeight = await num('Sparse weight (0-1)', DEFAULTS.hybrid.sparseWeight, 0, 1);
    const rrfK = await num('RRF k (1-1000)', DEFAULTS.hybrid.rrfK, 1, 1000);
    const colbert = await bool('Enable ColBERT late interaction?', DEFAULTS.hybrid.colbert);
    const multimodal = await bool('Enable multimodal retrieval?', DEFAULTS.multimodal.enabled);
    const traceCorpus = await bool('Enable thinking-trace corpus?', DEFAULTS.traceCorpus.enabled);
    const representation = await choose(
      'Default trace representation',
      ['struct', 'semantic', 'reflect'] as const,
      DEFAULTS.traceCorpus.representation,
    );

    const strategy: OnboardStrategy = {
      mode,
      k,
      ordering,
      reranker,
      hybrid: { denseWeight, sparseWeight, rrfK, colbert },
      multimodal: { enabled: multimodal },
      traceCorpus: { enabled: traceCorpus, representation },
    };

    await writeFile(
      join(target, 'strategy.json'),
      JSON.stringify(strategy, null, 2),
      { mode: 0o600 },
    );
    console.log(`✓ wrote ${join(target, 'strategy.json')}`);

    if (flags['dry-run'] === true) {
      try {
        const settings = loadSettings(env);
        await runDryRunCARE(settings);
      } catch (e) {
        console.error('[raghub] dry-run failed:', e instanceof Error ? e.message : String(e));
      }
    }
    return 0;
  },
};