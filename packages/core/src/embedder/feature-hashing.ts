/**
 * Deterministic feature-hashing embedder.
 *
 * No external call. Used in tests and as the fallback when no API key
 * is configured. Produces 3072-d vectors by default to match the
 * production sqlite-vec schema; tokens are hashed via xxhash-wasm
 * (would be a real lib in production — kept as a minimal hash for the
 * scaffold commit).
 */

import { ConfigurationError } from '../errors/index.js';
import type { Embedder } from './types.js';

const DEFAULT_DIM = 3072;

const fnv1a = (s: string): number => {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
};

const tokens = (text: string): readonly string[] =>
  text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 0);

export class FeatureHashingEmbedder implements Embedder {
  public readonly model: string;
  public readonly dimension: number;

  constructor(model: string = 'feature-hashing', dimension: number = DEFAULT_DIM) {
    if (dimension < 64 || dimension > 4096) {
      throw new ConfigurationError(`feature-hashing dimension out of range: ${dimension}`);
    }
    this.model = model;
    this.dimension = dimension;
  }

  public async embedQuery(text: string): Promise<readonly number[]> {
    return this.embedOne(text);
  }

  public async embedDocuments(texts: readonly string[]): Promise<readonly (readonly number[])[]> {
    return Promise.all(texts.map((t) => this.embedOne(t)));
  }

  private embedOne(text: string): readonly number[] {
    const vec = new Array<number>(this.dimension).fill(0);
    const toks = tokens(text);
    if (toks.length === 0) {
      return Object.freeze(vec.map(() => 0));
    }
    for (const tok of toks) {
      const h = fnv1a(tok);
      const idx = h % this.dimension;
      const sign = (h & 1) === 1 ? 1 : -1;
      vec[idx] = (vec[idx] ?? 0) + sign;
    }
    let norm = 0;
    for (const v of vec) norm += v * v;
    const denom = Math.sqrt(norm) || 1;
    return Object.freeze(vec.map((v) => v / denom));
  }
}