import { describe, expect, it } from 'vitest';

import {
  recallAtK,
  precisionAtK,
  mrr,
  faithfulness,
  answerCorrectness,
  computeMetrics,
} from '../src/metrics.js';
import { judgeCare, careMetrics } from '../src/care.js';
import { lostInMiddleProbe } from '../src/lost-in-middle.js';
import { aggregate, loadJsonl } from '../src/harness.js';
import { Chunk, ChunkModality, type Hit, brandId } from '@raghub/core';
import type { ChunkId, CollectionId, DocumentId, TenantId, UserId } from '@raghub/core';

const makeHit = (id: string, text: string): Hit => {
  const chunk = new Chunk({
    id: brandId<ChunkId>(id),
    tenantId: brandId<TenantId>('t'),
    ownerId: brandId<UserId>('u'),
    collectionId: brandId<CollectionId>('c'),
    documentId: brandId<DocumentId>('d'),
    modality: ChunkModality.Text,
    text,
    embedding: [],
    metadata: {},
    tokenCount: text.split(' ').length,
    createdAt: new Date(),
  });
  return { chunk, score: 1 };
};

describe('metrics', () => {
  it('recall_at_k counts ground-truth ids in the top k', () => {
    const r = recallAtK({
      hits: [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
      groundTruthIds: ['a', 'b', 'd'],
      k: 2,
    });
    expect(r).toBeCloseTo(2 / 3);
  });

  it('precision_at_k counts top-k hits over k', () => {
    const p = precisionAtK({
      hits: [{ id: 'a' }, { id: 'b' }, { id: 'x' }],
      groundTruthIds: ['a', 'b'],
      k: 3,
    });
    expect(p).toBeCloseTo(2 / 3);
  });

  it('mrr returns the reciprocal rank of the first ground-truth hit', () => {
    const r = mrr({
      hits: [{ id: 'x' }, { id: 'y' }, { id: 'a' }],
      groundTruthIds: ['a'],
    });
    expect(r).toBeCloseTo(1 / 3);
  });

  it('faithfulness counts claims supported by the context', () => {
    const f = faithfulness({
      hits: [],
      groundTruthIds: [],
      contextClaims: ['apple is red'],
      contextText: 'An apple is red and tasty.',
    });
    expect(f).toBeCloseTo(1);
  });

  it('answerCorrectness scores exact match as 1, partial as 0.5', () => {
    expect(answerCorrectness({ hits: [], groundTruthIds: [], answer: 'foo', reference: 'foo' })).toBe(1);
    expect(answerCorrectness({ hits: [], groundTruthIds: [], answer: 'foo bar', reference: 'foo' })).toBe(0.5);
    expect(answerCorrectness({ hits: [], groundTruthIds: [], answer: 'x', reference: 'y' })).toBe(0);
  });

  it('computeMetrics aggregates every metric at once', () => {
    const m = computeMetrics({
      hits: [{ id: 'a' }],
      groundTruthIds: ['a'],
      answer: 'apple',
      reference: 'apple',
      contextClaims: ['apple is red'],
      contextText: 'apple is red',
    });
    expect(m.recallAtK).toBeCloseTo(1);
    expect(m.precisionAtK).toBeCloseTo(1);
    expect(m.answerCorrectness).toBe(1);
  });
});

describe('CARE judge', () => {
  it('flags chunks that overlap the gold answer via the deterministic fallback', async () => {
    const labels = await judgeCare({
      list: [
        makeHit('a', 'apple is a fruit'),
        makeHit('b', 'bananas grow in tropical climates'),
      ],
      question: 'what is an apple',
      goldAnswer: 'apple is a fruit',
    });
    const map = new Map(labels.map((l) => [l.chunkId, l.relevant]));
    expect(map.get('a')).toBe(true);
    expect(map.get('b')).toBe(false);
  });

  it('careMetrics computes precision/recall/F1', () => {
    const m = careMetrics([
      { chunkId: 'a', relevant: true },
      { chunkId: 'b', relevant: false },
    ]);
    expect(m.precision).toBeCloseTo(0.5);
    expect(m.f1).toBeCloseTo(0.5);
  });
});

describe('lost-in-middle probe', () => {
  it('produces one sample per position in the list size', async () => {
    const samples = await lostInMiddleProbe({
      goldChunkIds: ['a'],
      candidateChunks: [makeHit('a', 'apple'), makeHit('b', 'banana')],
      listSize: 2,
      query: 'apple',
      reference: 'apple is a fruit',
    });
    expect(samples.length).toBe(2);
    expect(samples[0]?.position).toBe(0);
    expect(samples[1]?.position).toBe(1);
  });
});

describe('aggregate + loadJsonl', () => {
  it('aggregate averages every metric across samples', () => {
    const agg = aggregate([
      {
        sample: {
          id: '1',
          question: 'q',
          goldAnswer: 'a',
          goldIds: ['1'],
        },
        metrics: {
          recallAtK: 1,
          precisionAtK: 1,
          mrr: 1,
          faithfulness: 1,
          contextRecall: 1,
          contextPrecision: 1,
          answerCorrectness: 1,
        },
        latencyMs: 1,
        answer: 'a',
      },
      {
        sample: {
          id: '2',
          question: 'q',
          goldAnswer: 'a',
          goldIds: ['1'],
        },
        metrics: {
          recallAtK: 0,
          precisionAtK: 0,
          mrr: 0,
          faithfulness: 0,
          contextRecall: 0,
          contextPrecision: 0,
          answerCorrectness: 0,
        },
        latencyMs: 1,
        answer: 'x',
      },
    ]);
    expect(agg.count).toBe(2);
    expect(agg.recallAtK).toBe(0.5);
  });

  it('loadJsonl parses valid lines and skips malformed ones', () => {
    const samples = loadJsonl(
      '{"id":"1","question":"q","gold_answer":"a","gold_ids":["a"]}\nthis is not json\n{"id":"2","question":"q","gold_answer":"a","gold_ids":[]}\n',
    );
    expect(samples.length).toBe(2);
    expect(samples[0]?.id).toBe('1');
    expect(samples[1]?.id).toBe('2');
  });
});