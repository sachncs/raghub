/**
 * OpenAI embedder.
 *
 * Lazy-initialised; throws `ConfigurationError` if no API key is
 * present at the first call. Production deploys typically set
 * `OPENAI_API_KEY`; the embedder also accepts `RAGHUB_EMBEDDER_API_KEY`.
 */

import { ConfigurationError, MissingDepError } from '../errors/index.js';
import type { Embedder } from './types.js';

const OpenAI: typeof import('openai').default | undefined = (() => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return require('openai').default as typeof import('openai').default;
  } catch {
    return undefined;
  }
})();

export interface OpenAIEmbedderOptions {
  readonly model: string;
  readonly apiKey: string;
  readonly batchSize: number;
}

const DEFAULT_BATCH = 32;

export class OpenAIEmbedder implements Embedder {
  public readonly model: string;
  public readonly dimension: number;
  private readonly apiKey: string;
  private readonly batchSize: number;
  private client: import('openai').default | null = null;

  constructor(opts: OpenAIEmbedderOptions, dimension: number = 3072) {
    this.model = opts.model;
    this.dimension = dimension;
    this.apiKey = opts.apiKey;
    this.batchSize = opts.batchSize > 0 ? opts.batchSize : DEFAULT_BATCH;
  }

  public async embedQuery(text: string): Promise<readonly number[]> {
    const [vec] = await this.embedDocuments([text]);
    if (!vec) throw new ConfigurationError('openai embedder returned empty result');
    return vec;
  }

  public async embedDocuments(
    texts: readonly string[],
  ): Promise<readonly (readonly number[])[]> {
    const client = this.ensureClient();
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const slice = texts.slice(i, i + this.batchSize);
      const resp = await client.embeddings.create({
        model: this.model,
        input: [...slice],
      });
      for (const item of resp.data) {
        out.push([...item.embedding]);
      }
    }
    return Object.freeze(out.map((v) => Object.freeze(v)));
  }

  private ensureClient(): import('openai').default {
    if (this.client) return this.client;
    if (!OpenAI) {
      throw new MissingDepError('openai package is not installed; `pnpm add openai`', {
        details: { hint: 'pnpm add openai' },
      });
    }
    if (!this.apiKey) {
      throw new ConfigurationError('openai embedder requires an api key');
    }
    this.client = new OpenAI({ apiKey: this.apiKey });
    return this.client;
  }
}