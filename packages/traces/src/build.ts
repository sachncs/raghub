/**
 * Trace corpus builder.
 *
 * `buildTraceCorpus()` runs the full T3 pipeline for each input
 * problem:
 *
 *   1. thinker.generate(problem) -> raw trace
 *   2. transform(problem, raw) per representation (struct,
 *      semantic, reflect) — LLM-driven when available, otherwise
 *      the deterministic fallback in transforms.ts.
 *   3. embedder.embedQuery(representation_text) per representation
 *   4. SqliteTraceCorpus.insert() with all three representations
 *      + their embeddings.
 *
 * The output is reusable across inference: an offline thinker
 * populates it once; the orchestrator's trace_search tool reads
 * from it at query time.
 */

import {
  brandId,
  type Embedder,
  type Llm,
  type SqliteTraceCorpus,
  type WorkspaceId,
  type TraceId,
  type UserId,
  type TraceRepresentation,
} from '@raghub/core';

import { transformWithLlm } from './transforms.js';

export interface ProblemInput {
  readonly id?: string;
  readonly problem: string;
  readonly rawTrace?: string;
  readonly reference?: string;
}

export interface BuildOptions {
  readonly workspaceId: WorkspaceId;
  readonly userId?: UserId | null;
  readonly corpus: SqliteTraceCorpus;
  readonly embedder: Embedder;
  readonly thinker: Llm;
  readonly thinkerModel: string;
  readonly transformer?: Llm;
  readonly transformerModel?: string;
  readonly representations?: readonly TraceRepresentation[];
  readonly decontaminate?: (problem: string) => Promise<boolean>;
}

const ALL_REPS: readonly TraceRepresentation[] = ['struct', 'semantic', 'reflect'];

const generateRawTrace = async (
  problem: string,
  thinker: Llm,
  model: string,
): Promise<string> => {
  try {
    const result = await thinker.generate({
      model,
      temperature: 0,
      messages: [
        {
          role: 'system',
          content:
            'You are a precise reasoning model. Solve the problem step by step. Be concise but explicit about every reasoning step.',
        },
        { role: 'user', content: problem },
      ],
    });
    return result.content;
  } catch {
    return `No trace generated; problem was: ${problem}`;
  }
};

const buildId = (problem: string, raw: string): TraceId =>
  brandId<TraceId>(`trc_${rawHash(problem)}_${rawHash(raw.slice(0, 64))}`);

const rawHash = (s: string): string => {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(36);
};

export interface BuildResult {
  readonly inserted: number;
  readonly skipped: number;
}

export const buildTraceCorpus = async (
  inputs: readonly ProblemInput[],
  opts: BuildOptions,
): Promise<BuildResult> => {
  const reps = opts.representations ?? ALL_REPS;
  let inserted = 0;
  let skipped = 0;
  for (const input of inputs) {
    if (opts.decontaminate) {
      const reject = await opts.decontaminate(input.problem);
      if (reject) {
        skipped++;
        continue;
      }
    }
    const raw = input.rawTrace ?? (await generateRawTrace(input.problem, opts.thinker, opts.thinkerModel));
    const id = input.id ?? buildId(input.problem, raw);
    const transformer = opts.transformer ?? opts.thinker;
    const transformerModel = opts.transformerModel ?? opts.thinkerModel;

    const transformed: Partial<Record<TraceRepresentation, string>> = {};
    for (const rep of reps) {
      const out = await transformWithLlm(rep, raw, { llm: transformer, model: transformerModel });
      transformed[rep] = out;
    }

    const embedText = transformed.semantic ?? transformed.struct ?? raw;
    const embedding = await opts.embedder.embedQuery(embedText);

    const idTyped = id as unknown as TraceId;
    await opts.corpus.insert({
      id: idTyped,
      workspaceId: opts.workspaceId,
      userId: opts.userId ?? null,
      sourceProblem: input.problem,
      raw,
      struct: transformed.struct ?? null,
      semantic: transformed.semantic ?? null,
      reflect: transformed.reflect ?? null,
      embedding,
    });
    inserted++;
  }
  return { inserted, skipped };
};