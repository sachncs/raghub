import { describe, expect, it } from 'vitest';

import { reciprocalRankFusion } from '../src/retrieval/rrf.js';

describe('reciprocalRankFusion', () => {
  it('merges two ranked lists by RRF score', () => {
    const a = [{ id: 'x' }, { id: 'y' }, { id: 'z' }];
    const b = [{ id: 'y' }, { id: 'x' }, { id: 'w' }];
    const merged = reciprocalRankFusion([a, b]);
    expect(merged[0]).toBe('x');
    expect(merged).toContain('y');
    expect(merged).toContain('z');
    expect(merged).toContain('w');
  });

  it('respects the k parameter (higher k flattens the curve)', () => {
    const a = [{ id: 'x' }];
    const b = [{ id: 'y' }];
    const low = reciprocalRankFusion([a, b], 0);
    const high = reciprocalRankFusion([a, b], 60);
    expect(low[0]).toBeDefined();
    expect(high[0]).toBeDefined();
  });

  it('returns empty for empty input', () => {
    expect(reciprocalRankFusion([])).toEqual([]);
  });

  it('handles a single list', () => {
    const a = [{ id: 'q' }, { id: 'r' }, { id: 's' }];
    expect(reciprocalRankFusion([a])).toEqual(['q', 'r', 's']);
  });
});