import { describe, expect, it } from 'vitest';

import { brandId, type ChunkId, type CollectionId, type DocumentId, type UserId, type WorkspaceId } from '../src/domain/index.js';
import { Chunk } from '../src/domain/index.js';
import { agenticIngest } from '../src/ingest-agentic.js';
import type { VectorStore } from '../src/stores/index.js';
import type { GraphStore } from '../src/graph/store.js';
import type { WorkspaceMemoryStore, MemoryFact } from '../src/storage/memory.js';
import { MemoryScope } from '../src/storage/memory.js';

const wsp = brandId<WorkspaceId>('wsp_1');
const owner = brandId<UserId>('usr_owner');
const coll = brandId<CollectionId>('col_1');
const doc = brandId<DocumentId>('doc_1');

const text = Buffer.from('Microsoft was founded by Bill Gates. Apple was founded by Steve Jobs. RAG is retrieval-augmented generation.');

class FakeMemoryStore implements Pick<WorkspaceMemoryStore, 'remember'> {
  public facts: MemoryFact[] = [];
  public async remember(input: Parameters<WorkspaceMemoryStore['remember']>[0]): Promise<MemoryFact> {
    const f: MemoryFact = {
      id: this.facts.length + 1,
      workspaceId: input.workspaceId,
      scope: input.scope,
      userId: input.userId,
      content: input.content,
      metadata: { ...(input.metadata ?? {}) },
      createdAt: new Date(),
    };
    this.facts.push(f);
    return f;
  }
}

class FakeGraphStore implements Pick<GraphStore, 'addMentions' | 'searchEntities' | 'expandNeighborhood' | 'close'> {
  public mentions: { chunkId: ChunkId; entities: readonly string[] }[] = [];
  public async addMentions(workspaceId: WorkspaceId, chunkId: ChunkId, entities: readonly string[]): Promise<void> {
    void workspaceId;
    this.mentions.push({ chunkId, entities });
  }
  public async searchEntities(): Promise<never[]> {
    return [];
  }
  public async expandNeighborhood(): Promise<never[]> {
    return [];
  }
  public async close(): Promise<void> {
    // No-op.
  }
}

class FakeVectorStore implements Pick<VectorStore, 'add' | 'addBatch' | 'getById' | 'searchVector' | 'searchKeyword' | 'deleteByDocument' | 'close'> {
  public chunks: Chunk[] = [];
  public async add(chunk: Chunk): Promise<void> {
    this.chunks.push(chunk);
  }
  public async addBatch(chunks: readonly Chunk[]): Promise<void> {
    this.chunks.push(...chunks);
  }
  public async getById(_ws: WorkspaceId, id: ChunkId): Promise<Chunk | null> {
    return this.chunks.find((c) => c.id === id) ?? null;
  }
  public async searchVector(): Promise<never[]> {
    return [];
  }
  public async searchKeyword(): Promise<never[]> {
    return [];
  }
  public async deleteByDocument(): Promise<number> {
    return 0;
  }
  public async close(): Promise<void> {
    // No-op.
  }
}

describe('agenticIngest', () => {
  it('runs graph + memory steps in parallel after chunks land', async () => {
    const vec = new FakeVectorStore();
    const graph = new FakeGraphStore();
    const memory = new FakeMemoryStore();
    const embedder = {
      embedDocuments: async (texts: readonly string[]) => texts.map(() => new Array(8).fill(0.1)),
      embedQuery: async () => new Array(8).fill(0.1),
    };
    const result = await agenticIngest(
      {
        workspaceId: wsp,
        ownerId: owner,
        collectionId: coll,
        filename: 'a.txt',
        mimeType: 'text/plain',
        content: text,
      },
      {
        embedder,
        store: vec as never,
        extra: { graphStore: graph as never, memoryStore: memory as never },
      },
    );
    expect(result.alreadyExisted).toBe(false);
    expect(result.sideEffects.length).toBeGreaterThanOrEqual(1);
    const memoryEffect = result.sideEffects.find((s) => s.kind === 'memory');
    expect(memoryEffect?.kind).toBe('memory');
    expect(memory.facts.length).toBe(1);
    expect(memory.facts[0]?.content).toContain('a.txt');
  });

  it('does not crash without extra stores', async () => {
    const vec = new FakeVectorStore();
    const embedder = {
      embedDocuments: async (texts: readonly string[]) => texts.map(() => new Array(8).fill(0.1)),
      embedQuery: async () => new Array(8).fill(0.1),
    };
    const result = await agenticIngest(
      {
        workspaceId: wsp,
        ownerId: owner,
        collectionId: coll,
        filename: 'b.txt',
        mimeType: 'text/plain',
        content: Buffer.from('Plain text content with several sentences for chunking.'),
      },
      { embedder, store: vec as never },
    );
    expect(result.sideEffects.length).toBe(0);
  });

  it('returns alreadyExisted=true when seenHashes says so', async () => {
    const vec = new FakeVectorStore();
    const embedder = {
      embedDocuments: async () => [],
      embedQuery: async () => new Array(8).fill(0.1),
    };
    const result = await agenticIngest(
      {
        workspaceId: wsp,
        ownerId: owner,
        collectionId: coll,
        filename: 'c.txt',
        mimeType: 'text/plain',
        content: Buffer.from('hi'),
      },
      { embedder, store: vec as never, seenHashes: async () => true },
    );
    expect(result.alreadyExisted).toBe(true);
  });
});

void MemoryScope;