import { describe, expect, it } from 'vitest';

import { FeatureHashingEmbedder } from '../src/embedder/feature-hashing.js';
import { ConfigurationError } from '../src/errors/index.js';

describe('FeatureHashingEmbedder', () => {
  it('produces deterministic vectors for the same input', async () => {
    const e = new FeatureHashingEmbedder();
    const a = await e.embedQuery('hello world');
    const b = await e.embedQuery('hello world');
    expect(a).toEqual(b);
    expect(a.length).toBe(3072);
  });

  it('produces different vectors for different inputs', async () => {
    const e = new FeatureHashingEmbedder();
    const a = await e.embedQuery('hello world');
    const b = await e.embedQuery('goodbye sky');
    expect(a).not.toEqual(b);
  });

  it('returns an L2-normalised vector for non-empty input', async () => {
    const e = new FeatureHashingEmbedder();
    const v = await e.embedQuery('the quick brown fox');
    let n2 = 0;
    for (const x of v) n2 += x * x;
    expect(Math.abs(Math.sqrt(n2) - 1)).toBeLessThan(1e-6);
  });

  it('returns the zero vector for empty input', async () => {
    const e = new FeatureHashingEmbedder();
    const v = await e.embedQuery('   ');
    expect(v.every((x) => x === 0)).toBe(true);
  });

  it('embeds a batch and returns aligned dimensions', async () => {
    const e = new FeatureHashingEmbedder();
    const out = await e.embedDocuments(['x', 'y', 'z']);
    expect(out.length).toBe(3);
    for (const v of out) expect(v.length).toBe(3072);
  });

  it('rejects out-of-range dimension', () => {
    expect(() => new FeatureHashingEmbedder('m', 32)).toThrow(ConfigurationError);
  });

  it('exposes the configured model and dimension', () => {
    const e = new FeatureHashingEmbedder('custom-model', 1024);
    expect(e.model).toBe('custom-model');
    expect(e.dimension).toBe(1024);
  });
});