/**
 * Reranker registry + built-in implementations.
 *
 * Rerankers take a query and a list of hits and return a
 * re-ordered (or re-scored) version. The base package ships an
 * `IdentityReranker` (no-op) and a `CohereReranker` that calls
 * Cohere's `/v1/rerank` endpoint.
 *
 * Plug-in rerankers register via
 * `Registry.register('revex.rerankers', name, Ctor)`.
 */

import type { Hit } from '../domain/chunk.js';
import { Registry } from '../plugins/registry.js';

export interface Reranker {
  readonly name: string;
  rerank(query: string, hits: readonly Hit[], topK?: number): Promise<readonly Hit[]>;
}

export class IdentityReranker implements Reranker {
  readonly name = 'identity';
  async rerank(_q: string, hits: readonly Hit[]): Promise<readonly Hit[]> {
    return hits;
  }
}

export interface CohereRerankerOptions {
  readonly apiKey: string;
  readonly model?: string;
  readonly endpoint?: string;
}

export class CohereReranker implements Reranker {
  readonly name = 'cohere';
  private readonly apiKey: string;
  private readonly model: string;
  private readonly endpoint: string;

  constructor(opts: CohereRerankerOptions) {
    this.apiKey = opts.apiKey;
    this.model = opts.model ?? 'rerank-english-v3.0';
    this.endpoint = opts.endpoint ?? 'https://api.cohere.com/v1/rerank';
  }

  async rerank(query: string, hits: readonly Hit[], topK?: number): Promise<readonly Hit[]> {
    const documents = hits.map((h) => h.chunk.text);
    const res = await fetch(this.endpoint, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        query,
        documents,
        top_n: topK ?? documents.length,
        return_documents: false,
      }),
    });
    if (!res.ok) {
      return hits;
    }
    const body = (await res.json()) as {
      results?: { index: number; relevance_score: number }[];
    };
    const results = body.results ?? [];
    return results
      .slice(0, topK ?? results.length)
      .map((r) => {
        const hit = hits[r.index];
        if (!hit) return null;
        return { chunk: hit.chunk, score: r.relevance_score };
      })
      .filter((x): x is Hit => x !== null);
  }
}

export class LlmReranker implements Reranker {
  readonly name = 'llm';
  constructor(
    private readonly llm: { generate(opts: { model: string; temperature: number; messages: { role: string; content: string }[] }): Promise<{ content: string }> },
    private readonly model: string = 'gpt-4.1',
  ) {}

  async rerank(query: string, hits: readonly Hit[], topK = 5): Promise<readonly Hit[]> {
    if (hits.length === 0) return [];
    const docs = hits.slice(0, 20).map((h, i) => `[${i}] ${h.chunk.text.slice(0, 500)}`).join('\n\n');
    const r = await this.llm.generate({
      model: this.model,
      temperature: 0,
      messages: [
        {
          role: 'system',
          content: 'You are a relevance judge. Reply with a JSON array of document indices in descending relevance order.',
        },
        { role: 'user', content: `Query: ${query}\n\nDocuments:\n${docs}` },
      ],
    });
    try {
      const parsed = JSON.parse(r.content) as number[];
      return parsed
        .slice(0, topK)
        .map((i) => hits[i])
        .filter((h): h is Hit => h !== undefined);
    } catch {
      return hits.slice(0, topK);
    }
  }
}

export interface RerankerFactoryOptions {
  readonly kind: 'identity' | 'cohere' | 'llm';
  readonly apiKey?: string;
  readonly model?: string;
  readonly llm?: ConstructorParameters<typeof LlmReranker>[0];
}

export const RerankerFactory = (opts: RerankerFactoryOptions): Reranker => {
  switch (opts.kind) {
    case 'identity':
      return new IdentityReranker();
    case 'cohere':
      if (!opts.apiKey) throw new Error('cohere reranker requires apiKey');
      return new CohereReranker(
        opts.model !== undefined
          ? { apiKey: opts.apiKey, model: opts.model }
          : { apiKey: opts.apiKey },
      );
    case 'llm':
      if (!opts.llm) throw new Error('llm reranker requires llm');
      return opts.model !== undefined
        ? new LlmReranker(opts.llm, opts.model)
        : new LlmReranker(opts.llm);
  }
};

let registered = false;
export function registerBuiltInRerankers(): void {
  if (registered) return;
  Registry.register<IdentityReranker>('revex.rerankers', 'identity', IdentityReranker);
  Registry.register<CohereReranker>('revex.rerankers', 'cohere', CohereReranker);
  Registry.register<LlmReranker>('revex.rerankers', 'llm', LlmReranker);
  registered = true;
}