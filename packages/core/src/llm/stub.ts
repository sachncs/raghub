/**
 * Stub LLM — deterministic, network-free, delay-streaming.
 *
 * Selected when `RAGHUB_LLM_STUB=1` is set so the chat UI can be
 * exercised end-to-end (including the SSE streaming path) without
 * burning real tokens. Also useful in CI and for the agent-browser
 * smoke suite.
 *
 * Behaviour:
 *   - `provider` is `'stub'`; the requested model name is preserved
 *     so downstream logs show what would have been called.
 *   - `generate()` returns a predictable, short response derived
 *     from the last user message.
 *   - `stream()` yields the response word-by-word with a small
 *     delay so the UI's streaming render path is exercised.
 *   - `rawStream()` throws — stub has no byte-level wire format.
 */

import { ConfigurationError } from '../errors/index.js';
import type {
  ChatMessage,
  GenerateOptions,
  GenerateResult,
  Llm,
  StreamChunk,
} from './types.js';

const DEFAULT_DELAY_MS = 25;

export interface StubLlmOptions {
  readonly model: string;
  readonly delayMs?: number;
  readonly prefix?: string;
}

export class StubLlm implements Llm {
  public readonly provider = 'stub';
  public readonly model: string;
  private readonly delayMs: number;
  private readonly prefix: string;

  constructor(opts: StubLlmOptions) {
    this.model = opts.model;
    this.delayMs = opts.delayMs ?? DEFAULT_DELAY_MS;
    this.prefix = opts.prefix ?? 'STUB';
  }

  public async generate(opts: GenerateOptions): Promise<GenerateResult> {
    const text = buildAnswer(opts, this.prefix);
    return {
      content: text,
      toolCalls: [],
      usage: { promptTokens: 0, completionTokens: text.split(/\s+/).length, totalTokens: 0 },
      finishReason: 'stop',
    };
  }

  public async *stream(opts: GenerateOptions): AsyncIterable<StreamChunk> {
    const text = buildAnswer(opts, this.prefix);
    const tokens = text.split(/(\s+)/);
    for (const tok of tokens) {
      if (this.delayMs > 0) {
        await new Promise<void>((r) => setTimeout(r, this.delayMs));
      }
      yield { delta: tok, toolCalls: [], finishReason: null };
    }
    yield { delta: '', toolCalls: [], finishReason: 'stop' };
  }

  public async rawStream(_opts: GenerateOptions): Promise<never> {
    throw new ConfigurationError('StubLlm does not expose raw streams');
  }
}

const buildAnswer = (opts: GenerateOptions, prefix: string): string => {
  const lastUser = [...opts.messages].reverse().find((m) => m.role === 'user');
  const q = (lastUser?.content ?? '').trim();
  const truncated = q.length > 120 ? `${q.slice(0, 120)}…` : q;
  return `${prefix}[${opts.model}]: received "${truncated || '(empty)'}".`;
};