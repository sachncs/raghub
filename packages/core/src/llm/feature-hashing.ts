/**
 * FeatureHashing LLM stub.
 *
 * The default no-network fallback for tests and sandboxed environments.
 * Produces deterministic, non-ML answers by replaying context into a
 * "summary" format. It is NOT a substitute for the real LLM; it exists
 * so the orchestrator runs without an API key and so test suites stay
 * green in CI.
 */

import { ConfigurationError } from '../errors/index.js';
import type { ChatMessage, GenerateOptions, GenerateResult, Llm, StreamChunk } from './types.js';

export class FeatureHashingLlm implements Llm {
  public readonly provider = 'feature-hashing';
  public readonly model: string;

  constructor(model: string = 'feature-hashing-llm') {
    this.model = model;
  }

  public async generate(opts: GenerateOptions): Promise<GenerateResult> {
    const text = buildAnswer(opts);
    return {
      content: text,
      toolCalls: [],
      usage: { promptTokens: countTokens(opts.messages), completionTokens: countTokens(text), totalTokens: 0 },
      finishReason: 'stop',
    };
  }

  public async *stream(opts: GenerateOptions): AsyncIterable<StreamChunk> {
    const text = buildAnswer(opts);
    const words = text.split(/(\s+)/);
    for (const w of words) {
      yield { delta: w, toolCalls: [], finishReason: null };
    }
    yield { delta: '', toolCalls: [], finishReason: 'stop' };
  }

  public async rawStream(): Promise<never> {
    throw new ConfigurationError('FeatureHashingLlm does not expose raw streams');
  }
}

const buildAnswer = (opts: GenerateOptions): string => {
  const lastUser = [...opts.messages].reverse().find((m) => m.role === 'user');
  const q = lastUser?.content ?? '';
  const ctx = opts.messages
    .filter((m) => m.role === 'system')
    .map((m: ChatMessage) => m.content)
    .join('\n');
  if (ctx.trim().length === 0) return `Echo: ${q}`;
  const preview = ctx.length > 600 ? `${ctx.slice(0, 600)}…` : ctx;
  return `Based on the provided context:\n\n${preview}\n\nQuestion: ${q}\n\nAnswer: the evidence above is the most relevant match in this corpus.`;
};

const countTokens = (text: string | readonly ChatMessage[]): number => {
  if (typeof text === 'string') return text.split(/\s+/).length;
  return text.reduce((n, m) => n + m.content.split(/\s+/).length, 0);
};