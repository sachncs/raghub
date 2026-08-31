import { describe, expect, it } from 'vitest';

import {
  buildContext,
  defaultBudget,
  summariseContext,
  type ContextBudget,
  type ContextHit,
} from '../../src/index.js';

const hit = (id: string, score: number, text: string, documentId?: string): ContextHit => ({
  id,
  score,
  text,
  ...(documentId !== undefined ? { documentId } : {}),
});

describe('buildContext', () => {
  it('assembles system + user when no hits and no history', () => {
    const result = buildContext({
      question: 'What is hybrid retrieval?',
      hits: [],
      systemPrompt: 'You are Revex.',
      budget: defaultBudget('gpt-4o'),
    });
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]?.role).toBe('system');
    expect(result.messages[1]?.role).toBe('user');
    expect(result.messages[1]?.content).toBe('What is hybrid retrieval?');
    expect(result.stats.hitsIncluded).toBe(0);
    expect(result.stats.historyTurnsIncluded).toBe(0);
  });

  it('includes hits ordered by score (highest first)', () => {
    const result = buildContext({
      question: 'q',
      hits: [
        hit('a', 0.5, 'low'),
        hit('b', 0.9, 'high'),
        hit('c', 0.7, 'mid'),
      ],
      systemPrompt: 'sys',
      budget: defaultBudget('gpt-4o'),
    });
    const sys = result.messages[0]?.content ?? '';
    const idxB = sys.indexOf('high');
    const idxC = sys.indexOf('mid');
    const idxA = sys.indexOf('low');
    expect(idxB).toBeGreaterThan(-1);
    expect(idxC).toBeGreaterThan(idxB);
    expect(idxA).toBeGreaterThan(idxC);
  });

  it('drops the lowest-scoring hits when over budget', () => {
    const big = 'a'.repeat(40_000); // ~10k tokens
    const huge = defaultBudget('nano');
    const result = buildContext({
      question: 'q',
      hits: [
        hit('top', 0.9, big, 'doc-top'),
        hit('mid', 0.5, big, 'doc-mid'),
        hit('low', 0.1, big, 'doc-low'),
      ],
      systemPrompt: 'sys',
      budget: huge,
    });
    expect(result.stats.hitsIncluded).toBeGreaterThanOrEqual(1);
    expect(result.stats.hitsTruncated).toBeGreaterThanOrEqual(1);
    const sys = result.messages[0]?.content ?? '';
    expect(sys).toContain('top');
    expect(sys).not.toContain('low');
  });

  it('truncates oversized hits with a [truncated] marker', () => {
    const oversized = 'b'.repeat(60_000);
    const result = buildContext({
      question: 'q',
      hits: [hit('only', 0.99, oversized, 'doc-1')],
      systemPrompt: 'sys',
      budget: defaultBudget('nano'),
    });
    const sys = result.messages[0]?.content ?? '';
    expect(sys).toContain('[truncated]');
    expect(sys.length).toBeLessThan(oversized.length);
  });

  it('drops the oldest history turns first when history overflows', () => {
    const budget: ContextBudget = {
      ...defaultBudget('nano'),
      historyMaxTokens: 200,
    };
    const result = buildContext({
      question: 'q',
      hits: [],
      history: [
        { role: 'user', content: 'old turn 1'.repeat(50) },
        { role: 'assistant', content: 'old turn 2'.repeat(50) },
        { role: 'user', content: 'recent turn' },
      ],
      systemPrompt: 'sys',
      budget,
    });
    expect(result.stats.historyTurnsIncluded).toBeGreaterThanOrEqual(1);
    expect(result.stats.historyTurnsDropped).toBeGreaterThanOrEqual(1);
    const lastMsg = result.messages[result.messages.length - 2];
    expect(lastMsg?.content).toBe('recent turn');
  });

  it('coalesces adjacent chunks from the same document', () => {
    const mid = 'c'.repeat(16_000);
    const tightBudget: ContextBudget = {
      ...defaultBudget('nano'),
      retrievalMaxTokens: 8_500,
    };
    const result = buildContext({
      question: 'q',
      hits: [
        hit('a', 0.9, mid, 'doc-same'),
        hit('b', 0.8, mid, 'doc-same'),
        hit('c', 0.7, mid, 'doc-other'),
      ],
      systemPrompt: 'sys',
      budget: tightBudget,
    });
    expect(result.stats.coalescedChunks).toBeGreaterThanOrEqual(1);
    const sys = result.messages[0]?.content ?? '';
    expect(sys).toContain('[doc-same]');
    expect(sys).toContain('[doc-other]');
  });

  it('respects the total budget ceiling', () => {
    const result = buildContext({
      question: 'q',
      hits: Array.from({ length: 10 }, (_, i) => hit(`h${i}`, 1 - i * 0.05, 'x'.repeat(2_000))),
      history: Array.from({ length: 20 }, () => ({ role: 'user' as const, content: 'y'.repeat(1_000) })),
      systemPrompt: 'sys',
      budget: defaultBudget('nano'),
    });
    const ceiling =
      result.stats.systemTokens +
      result.stats.retrievalTokens +
      result.stats.historyTokens +
      result.stats.userTokens;
    expect(ceiling).toBe(result.stats.totalTokens);
    expect(result.stats.retrievalTokens).toBeLessThanOrEqual(
      defaultBudget('nano').retrievalMaxTokens,
    );
    expect(result.stats.historyTokens).toBeLessThanOrEqual(
      defaultBudget('nano').historyMaxTokens,
    );
  });
});

describe('summariseContext', () => {
  it('renders a compact summary', () => {
    const result = buildContext({
      question: 'q',
      hits: [hit('a', 0.9, 'x')],
      systemPrompt: 'sys',
      budget: defaultBudget('gpt-4o'),
    });
    const summary = summariseContext(result.stats);
    expect(summary).toContain('hits');
    expect(summary).toContain('turns');
  });
});

describe('defaultBudget', () => {
  it('uses a small budget for nano models', () => {
    const b = defaultBudget('gpt-4o-mini');
    expect(b.totalTokens).toBeLessThan(64_000);
  });

  it('uses a large budget for flagship models', () => {
    const b = defaultBudget('gpt-4o');
    expect(b.totalTokens).toBeGreaterThanOrEqual(100_000);
  });
});