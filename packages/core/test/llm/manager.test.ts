/**
 * LlmManager unit tests — retry, fallback, usage aggregation.
 *
 * Uses fake Llm implementations to drive deterministic retry +
 * fallback behavior. Network jitter is stubbed so the test
 * finishes in <100ms.
 */

import { describe, expect, it, vi } from 'vitest';

import {
  LlmError,
  LlmManager,
  classifyError,
  estimateTokens,
  estimateMessagesTokens,
} from '../../src/index.js';
import type { Llm, GenerateOptions, GenerateResult, StreamChunk } from '../../src/index.js';

const fakeResult = (text = 'hello'): GenerateResult => ({
  content: text,
  toolCalls: [],
  usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 },
  finishReason: 'stop',
});

const okLlm = (provider: string, model = 'ok-model'): Llm => ({
  provider,
  model,
  async generate() {
    return fakeResult(`${provider}::answer`);
  },
  async *stream() {
    yield { delta: `${provider}::chunk`, toolCalls: [], finishReason: null };
    yield { delta: '', toolCalls: [], finishReason: 'stop' };
  },
  async rawStream() {
    throw new Error('not used in tests');
  },
});

const flakyLlm = (
  provider: string,
  failuresBeforeSuccess: number,
  model = 'flaky',
): { llm: Llm; calls: { count: number } } => {
  const counter = { count: 0 };
  const llm: Llm = {
    provider,
    model,
    async generate() {
      counter.count += 1;
      if (counter.count <= failuresBeforeSuccess) {
        throw new Error('rate limit exceeded');
      }
      return fakeResult(`${provider}::recovered`);
    },
    async *stream() {
      counter.count += 1;
      if (counter.count <= failuresBeforeSuccess) {
        throw new Error('rate limit exceeded');
      }
      yield { delta: 'recovered', toolCalls: [], finishReason: 'stop' };
    },
    async rawStream() {
      throw new Error('not used');
    },
  };
  return { llm, calls: counter };
};

const neverWorks = (provider: string): Llm => ({
  provider,
  model: 'broken',
  async generate() {
    const err = new Error('internal server error') as Error & { status?: number };
    err.status = 503;
    throw err;
  },
  async *stream() {
    const err = new Error('internal server error') as Error & { status?: number };
    err.status = 503;
    throw err;
  },
  async rawStream() {
    throw new Error('not used');
  },
});

const fastSleep = (): void => {};

describe('LlmManager', () => {
  it('returns the primary result on success and records usage', async () => {
    const primary = okLlm('openai');
    const manager = new LlmManager({
      primary,
      fallbacks: [okLlm('stub')],
      maxAttempts: 3,
      baseDelayMs: 1,
      maxDelayMs: 10,
      onAttempt: fastSleep,
    });
    const result = await manager.generate({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'hi' }],
    });
    expect(result.content).toBe('openai::answer');
    const u = manager.usage();
    expect(u.attempts).toBe(1);
    expect(u.failures).toBe(0);
    expect(u.fallbacksUsed).toBe(0);
    expect(u.promptTokens).toBe(10);
    expect(u.completionTokens).toBe(5);
    expect(u.totalTokens).toBe(15);
    expect(u.providers).toEqual([{ provider: 'openai', calls: 1 }]);
  });

  it('retries transient errors and eventually succeeds', async () => {
    const { llm: primary, calls } = flakyLlm('openai', 2);
    const manager = new LlmManager({
      primary,
      maxAttempts: 5,
      baseDelayMs: 1,
      maxDelayMs: 5,
      onAttempt: fastSleep,
    });
    const result = await manager.generate({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'retry me' }],
    });
    expect(result.content).toBe('openai::recovered');
    expect(calls.count).toBe(3);
    const u = manager.usage();
    expect(u.failures).toBe(2);
    expect(u.attempts).toBe(3);
    expect(u.fallbacksUsed).toBe(0);
  });

  it('falls back to the next provider after exhausting retries', async () => {
    const primary = neverWorks('openai');
    const fallback = okLlm('stub');
    const manager = new LlmManager({
      primary,
      fallbacks: [fallback],
      maxAttempts: 2,
      baseDelayMs: 1,
      maxDelayMs: 5,
      onAttempt: fastSleep,
    });
    const result = await manager.generate({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'fallback me' }],
    });
    expect(result.content).toBe('stub::answer');
    const u = manager.usage();
    expect(u.fallbacksUsed).toBe(1);
    expect(u.providers.map((p) => p.provider)).toContain('openai');
    expect(u.providers.map((p) => p.provider)).toContain('stub');
  });

  it('surfaces non-retryable errors immediately', async () => {
    const authErrorLlm: Llm = {
      provider: 'openai',
      model: 'gpt-4o',
      async generate() {
        throw new Error('401 unauthorized: api key invalid');
      },
      async *stream() {
        throw new Error('401 unauthorized: api key invalid');
      },
      async rawStream() {
        throw new Error('not used');
      },
    };
    const manager = new LlmManager({
      primary: authErrorLlm,
      fallbacks: [okLlm('stub')],
      maxAttempts: 3,
      baseDelayMs: 1,
      maxDelayMs: 5,
      onAttempt: fastSleep,
    });
    await expect(
      manager.generate({ model: 'gpt-4o', messages: [{ role: 'user', content: 'x' }] }),
    ).rejects.toBeInstanceOf(LlmError);
    const u = manager.usage();
    expect(u.fallbacksUsed).toBe(0);
    expect(u.failures).toBe(1);
  });

  it('streams through fallbacks', async () => {
    const primary = neverWorks('openai');
    const fallback = okLlm('stub');
    const manager = new LlmManager({
      primary,
      fallbacks: [fallback],
      maxAttempts: 1,
      baseDelayMs: 1,
      maxDelayMs: 5,
      onAttempt: fastSleep,
    });
    const out: StreamChunk[] = [];
    for await (const chunk of manager.stream({
      model: 'stub-model',
      messages: [{ role: 'user', content: 'hi' }],
    })) {
      out.push(chunk);
    }
    expect(out.map((c) => c.delta).join('')).toBe('stub::chunk');
  });
});

describe('classifyError', () => {
  it('flags rate-limit messages as retryable', () => {
    const e = classifyError('openai', new Error('rate limit exceeded'));
    expect(e.kind).toBe('rate_limit');
    expect(e.retryable).toBe(true);
  });

  it('flags auth errors as terminal', () => {
    const e = classifyError('openai', new Error('401 unauthorized'));
    expect(e.kind).toBe('auth');
    expect(e.retryable).toBe(false);
  });

  it('flags context overflow as terminal', () => {
    const e = classifyError('openai', new Error('maximum context length exceeded'));
    expect(e.kind).toBe('context_overflow');
    expect(e.retryable).toBe(false);
  });

  it('returns an LlmError unchanged', () => {
    const original = new LlmError({
      kind: 'auth',
      provider: 'openai',
      message: 'bad key',
      retryable: false,
    });
    expect(classifyError('openai', original)).toBe(original);
  });
});

describe('estimateTokens', () => {
  it('rounds up by length/4', () => {
    expect(estimateTokens('')).toBe(1);
    expect(estimateTokens('hello world')).toBe(3);
    expect(estimateTokens('a'.repeat(400))).toBe(100);
  });

  it('sums message tokens plus a per-message overhead', () => {
    const total = estimateMessagesTokens([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
    ]);
    expect(total).toBeGreaterThan(0);
  });
});