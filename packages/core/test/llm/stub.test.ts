/**
 * StubLlm — deterministic, network-free, delay-streaming.
 *
 * Selected when `RAGHUB_LLM_STUB=1`. Verifies the chunked-stream
 * contract so the SSE proxy / chat UI can rely on it.
 */

import { describe, expect, it } from 'vitest';

import { StubLlm } from '../../src/llm/stub.js';

const messages = (text: string) => [
  { role: 'system' as const, content: 'You are a helpful assistant.' },
  { role: 'user' as const, content: text },
];

describe('StubLlm', () => {
  it('exposes the requested model and provider "stub"', () => {
    const llm = new StubLlm({ model: 'gpt-4.1' });
    expect(llm.provider).toBe('stub');
    expect(llm.model).toBe('gpt-4.1');
  });

  it('generate() returns a deterministic, prefix-tagged response', async () => {
    const llm = new StubLlm({ model: 'm', prefix: 'TAG' });
    const result = await llm.generate({ model: 'm', messages: messages('hello world') });
    expect(result.finishReason).toBe('stop');
    expect(result.content).toContain('TAG[m]');
    expect(result.content).toContain('hello world');
  });

  it('stream() yields per-token chunks then a final stop', async () => {
    const llm = new StubLlm({ model: 'm', delayMs: 0 });
    const chunks: string[] = [];
    for await (const c of llm.stream({ model: 'm', messages: messages('a b c') })) {
      chunks.push(c.delta);
    }
    const text = chunks.join('');
    expect(text).toContain('a b c');
    expect(chunks[chunks.length - 1]).toBe('');
  });

  it('stream() respects an explicit delay', async () => {
    const llm = new StubLlm({ model: 'm', delayMs: 5 });
    const start = Date.now();
    let count = 0;
    for await (const _ of llm.stream({ model: 'm', messages: messages('one two three four') })) {
      count++;
    }
    expect(count).toBeGreaterThan(4);
    expect(Date.now() - start).toBeGreaterThanOrEqual(20);
  });

  it('truncates long inputs in the echoed answer', async () => {
    const llm = new StubLlm({ model: 'm' });
    const long = 'x'.repeat(500);
    const result = await llm.generate({ model: 'm', messages: messages(long) });
    expect(result.content).toContain('…');
    expect(result.content.length).toBeLessThan(long.length);
  });

  it('handles an empty user message', async () => {
    const llm = new StubLlm({ model: 'm' });
    const result = await llm.generate({ model: 'm', messages: [{ role: 'user', content: '' }] });
    expect(result.content).toContain('(empty)');
  });

  it('rawStream() throws', async () => {
    const llm = new StubLlm({ model: 'm' });
    await expect(llm.rawStream({ model: 'm', messages: messages('x') })).rejects.toThrow(
      /does not expose raw streams/,
    );
  });
});