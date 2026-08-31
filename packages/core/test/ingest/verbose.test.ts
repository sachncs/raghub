/**
 * Verbose ingest tests.
 *
 * Drives `ingestVerbose` with a fake embedder + vector store so the
 * lifecycle events fire deterministically. Asserts the timeline
 * matches the contract documented in `ingest-verbose.ts`.
 */

import { describe, expect, it } from 'vitest';

import { IngestEmitter, ingestVerbose } from '../../src/index.js';
import type { Embedder } from '../../src/index.js';
import type { StoreStats, VectorStore } from '../../src/index.js';
import { brandId } from '../../src/domain/index.js';

class FakeEmbedder implements Embedder {
  public readonly model = 'fake';
  public readonly dimension = 4;
  async embedQuery(): Promise<readonly number[]> {
    return [0.1, 0.2, 0.3, 0.4];
  }
  async embedDocuments(texts: readonly string[]): Promise<readonly (readonly number[])[]> {
    return texts.map(() => [0.1, 0.2, 0.3, 0.4]);
  }
}

class FakeStore implements VectorStore {
  public addedBatches: { chunkCount: number }[] = [];
  async add(): Promise<void> {}
  async addBatch(chunks: readonly unknown[]): Promise<void> {
    this.addedBatches.push({ chunkCount: chunks.length });
  }
  async searchVector() {
    return [];
  }
  async searchKeyword() {
    return [];
  }
  async getById() {
    return null;
  }
  async deleteByDocument() {
    return 0;
  }
  async stats(): Promise<StoreStats> {
    return {
      documentCount: 0,
      chunkCount: 0,
      embeddingBytes: 0,
      totalTokenEstimate: 0,
      bytesOnDisk: 0,
      lastIngestedAt: null,
      statusCounts: {},
    };
  }
  async close(): Promise<void> {}
}

describe('ingestVerbose', () => {
  it('emits the full lifecycle for a fresh ingest', async () => {
    const emitter = new IngestEmitter();
    const events: string[] = [];
    emitter.on((e) => events.push(e.phase));

    const store = new FakeStore();
    const output = await ingestVerbose(
      {
        workspaceId: brandId('wsp_x'),
        ownerId: brandId('usr_x'),
        collectionId: brandId('col_x'),
        filename: 'note.txt',
        mimeType: 'text/plain',
        content: Buffer.from('Revex is a hybrid retrieval engine for teams. '),
      },
      { embedder: new FakeEmbedder(), store },
      { emitter, batchSize: 8 },
    );

    expect(events).toEqual([
      'start',
      'parsed',
      'chunked',
      'embedding',
      'persisting',
      'indexed',
    ]);
    expect(output.alreadyExisted).toBe(false);
    expect(store.addedBatches.length).toBe(1);
  });

  it('emits a skipped event when the hash is already indexed', async () => {
    const emitter = new IngestEmitter();
    const events: string[] = [];
    emitter.on((e) => events.push(e.phase));

    const store = new FakeStore();
    const buffer = Buffer.from('already-indexed content');
    const output = await ingestVerbose(
      {
        workspaceId: brandId('wsp_x'),
        ownerId: brandId('usr_x'),
        collectionId: brandId('col_x'),
        filename: 'dup.txt',
        mimeType: 'text/plain',
        content: buffer,
      },
      {
        embedder: new FakeEmbedder(),
        store,
        seenHashes: async () => true,
      },
      { emitter, batchSize: 8 },
    );

    expect(events).toEqual(['start', 'skipped']);
    expect(output.alreadyExisted).toBe(true);
  });

  it('reports multiple embedding batches when chunk count exceeds batchSize', async () => {
    const emitter = new IngestEmitter();
    const batchSizes: number[] = [];
    emitter.on((e) => {
      if (e.phase === 'embedding') batchSizes.push(e.batchSize);
    });

    const store = new FakeStore();
    await ingestVerbose(
      {
        workspaceId: brandId('wsp_x'),
        ownerId: brandId('usr_x'),
        collectionId: brandId('col_x'),
        filename: 'big.txt',
        mimeType: 'text/plain',
        content: Buffer.from('a '.repeat(20_000)),
      },
      { embedder: new FakeEmbedder(), store },
      { emitter, batchSize: 4 },
    );

    expect(batchSizes.length).toBeGreaterThan(1);
    expect(batchSizes.every((n) => n <= 8)).toBe(true);
  });

  it('emits failed event when embedder fails mid-batch', async () => {
    const emitter = new IngestEmitter();
    const events: string[] = [];
    emitter.on((e) => events.push(e.phase));

    const failingEmbedder: Embedder = {
      model: 'fake',
      dimension: 4,
      async embedQuery() {
        return [];
      },
      async embedDocuments() {
        throw new Error('boom');
      },
    };

    await expect(
      ingestVerbose(
        {
          workspaceId: brandId('wsp_x'),
          ownerId: brandId('usr_x'),
          collectionId: brandId('col_x'),
          filename: 'broken.txt',
          mimeType: 'text/plain',
          content: Buffer.from('a'.repeat(2_000)),
        },
        { embedder: failingEmbedder, store: new FakeStore() },
        { emitter, batchSize: 8 },
      ),
    ).rejects.toThrow('boom');
    expect(events).toContain('start');
    expect(events).toContain('parsed');
    expect(events).toContain('chunked');
  });

  it('supports unsubscribe', () => {
    const emitter = new IngestEmitter();
    const seen: string[] = [];
    const off = emitter.on((e) => seen.push(e.phase));
    emitter.emit({ phase: 'start', filename: '', byteSize: 0, mimeType: '', hash: '', documentId: '' });
    expect(seen.length).toBe(1);
    off();
    emitter.emit({ phase: 'parsed', kind: 'text', characterCount: 0, parserLatencyMs: 0, metadata: {} });
    expect(seen.length).toBe(1);
  });
});