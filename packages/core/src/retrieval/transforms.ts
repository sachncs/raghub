/**
 * Query-side transformations for advanced RAG.
 *
 * Each transformer takes a question + the configured collaborators
 * and returns one or more expanded queries. The retrieval
 * pipeline can then fan out and fuse.
 *
 * - `HydeTransformer`: asks the LLM for a hypothetical answer
 *   passage, embeds it, and uses the embedding as the query.
 * - `MultiQueryTransformer`: asks for N alternate phrasings, then
 *   the caller runs retrieval on every variant.
 * - `StepBackTransformer`: asks for an abstracted ("step-back")
 *   question that targets the underlying concept.
 * - `DecomposeTransformer`: splits a multi-hop question into
 *   sub-questions.
 * - `ComposeTransformer`: takes many sub-results and produces a
 *   single coherent answer.
 */

import type { Llm } from '../llm/types.js';

const HYDE_PROMPT = (q: string): string =>
  'Write a short, plausible passage (3-5 sentences) that would answer the following question. ' +
  'Do not hedge or say you do not know — produce the passage as if it were real.\n\nQuestion: ' + q;

const MULTI_QUERY_PROMPT = (q: string, n: number): string =>
  `Generate ${n} different phrasings of the following question that would retrieve the same information. ` +
  'Reply with one phrasing per line, no numbering, no commentary.\n\nQuestion: ' + q;

const STEP_BACK_PROMPT = (q: string): string =>
  'Given the specific question below, write one more abstract / higher-level question that would retrieve the ' +
  'principle or concept needed to answer it. Reply with only the abstracted question, no commentary.\n\nQuestion: ' + q;

const DECOMPOSE_PROMPT = (q: string): string =>
  'Split the following multi-hop question into atomic sub-questions. Reply with one sub-question per line, ' +
  'no numbering, no commentary. If the question is already atomic, reply with just the question.\n\nQuestion: ' + q;

const COMPOSE_SYSTEM =
  'You are a precise synthesiser. You will be given a question and several retrieved passages. Produce a ' +
  'concise answer (3-6 sentences) that uses only facts present in the passages. Cite passage numbers in square ' +
  'brackets like [1], [2]. If the passages do not contain the answer, say so.';

export interface HydeTransformer {
  readonly name: 'hyde';
  expand(question: string): Promise<string>;
}

export interface MultiQueryTransformer {
  readonly name: 'multi_query';
  expand(question: string, n?: number): Promise<readonly string[]>;
}

export interface StepBackTransformer {
  readonly name: 'step_back';
  expand(question: string): Promise<string>;
}

export interface DecomposeTransformer {
  readonly name: 'decompose';
  expand(question: string): Promise<readonly string[]>;
}

export interface ComposeTransformer {
  readonly name: 'compose';
  compose(question: string, passages: readonly { readonly id: number; readonly text: string }[]): Promise<string>;
}

export const createHydeTransformer = (llm: Llm): HydeTransformer => ({
  name: 'hyde',
  async expand(question: string): Promise<string> {
    const r = await llm.generate({
      model: llm.model,
      temperature: 0.7,
      messages: [{ role: 'user', content: HYDE_PROMPT(question) }],
    });
    return r.content.trim();
  },
});

export const createMultiQueryTransformer = (llm: Llm): MultiQueryTransformer => ({
  name: 'multi_query',
  async expand(question: string, n = 4): Promise<readonly string[]> {
    const r = await llm.generate({
      model: llm.model,
      temperature: 0.5,
      messages: [{ role: 'user', content: MULTI_QUERY_PROMPT(question, n) }],
    });
    return r.content
      .split(/\n+/)
      .map((s) => s.replace(/^[-*\d.\s]+/, '').trim())
      .filter((s) => s.length > 0);
  },
});

export const createStepBackTransformer = (llm: Llm): StepBackTransformer => ({
  name: 'step_back',
  async expand(question: string): Promise<string> {
    const r = await llm.generate({
      model: llm.model,
      temperature: 0,
      messages: [{ role: 'user', content: STEP_BACK_PROMPT(question) }],
    });
    return r.content.trim();
  },
});

export const createDecomposeTransformer = (llm: Llm): DecomposeTransformer => ({
  name: 'decompose',
  async expand(question: string): Promise<readonly string[]> {
    const r = await llm.generate({
      model: llm.model,
      temperature: 0,
      messages: [{ role: 'user', content: DECOMPOSE_PROMPT(question) }],
    });
    return r.content
      .split(/\n+/)
      .map((s) => s.replace(/^[-*\d.\s]+/, '').trim())
      .filter((s) => s.length > 0);
  },
});

export const createComposeTransformer = (llm: Llm): ComposeTransformer => ({
  name: 'compose',
  async compose(question, passages) {
    if (passages.length === 0) {
      return `I could not find relevant information to answer: ${question}`;
    }
    const blocks = passages.map((p, i) => `[${i + 1}] ${p.text}`).join('\n\n');
    const r = await llm.generate({
      model: llm.model,
      temperature: 0,
      messages: [
        { role: 'system', content: COMPOSE_SYSTEM },
        { role: 'user', content: `Question: ${question}\n\nPassages:\n${blocks}` },
      ],
    });
    return r.content.trim();
  },
});

export const CascadeStages = ['cheap', 'medium', 'expensive'] as const;
export type CascadeStage = (typeof CascadeStages)[number];

export interface CascadeRouter {
  readonly name: 'cascade';
  route(question: string): CascadeStage;
}

export const createCascadeRouter = (opts?: { readonly cheap?: RegExp[] }): CascadeRouter => ({
  name: 'cascade',
  route(question: string): CascadeStage {
    const cheap = opts?.cheap ?? [
      /^\s*who\s+is\b/i,
      /^\s*what\s+is\b/i,
      /^\s*when\s+did\b/i,
      /^\s*define\b/i,
    ];
    if (cheap.some((re) => re.test(question))) return 'cheap';
    if (question.length > 200) return 'expensive';
    return 'medium';
  },
});