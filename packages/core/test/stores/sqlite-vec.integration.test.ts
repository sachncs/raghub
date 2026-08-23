import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import {
  Chunk,
  ChunkModality,
  User,
  UserRole,
  brandId,
} from '../../src/domain/index.js';
import type {
  ChunkId,
  CollectionId,
  DocumentId,
  TenantId,
  UserId,
} from '../../src/domain/index.js';
import { FeatureHashingEmbedder } from '../../src/embedder/feature-hashing.js';
import { Retrieval } from '../../src/retrieval/pipeline.js';
import { SqliteVecStore } from '../../src/stores/sqlite-vec.js';

const tenant = brandId<TenantId>('tnt_1');
const user = brandId<UserId>('usr_1');
const coll = brandId<CollectionId>('col_1');
const doc = brandId<DocumentId>('doc_1');

const mkUser = (role: 'admin' | 'member' | 'viewer') =>
  new User({
    id: user,
    tenantId: tenant,
    email: 'u@x',
    role: role === 'admin' ? UserRole.Admin : role === 'member' ? UserRole.Member : UserRole.Viewer,
    allowedCompanies: role === 'admin' ? [] : ['acme'],
    createdAt: new Date(),
  });

const seed = async (store: SqliteVecStore, embedder: FeatureHashingEmbedder) => {
  const chunks = [
    'raghub is a retrieval-augmented generation framework',
    'Strands Agents is an SDK for multi-agent orchestration',
    'bm25 is a sparse retrieval algorithm',
    'sqlite-vec is a vector search extension for SQLite',
  ];
  for (const text of chunks) {
    const v = await embedder.embedQuery(text);
    const c = new Chunk({
      id: brandId<ChunkId>(`chk_${Math.random().toString(36).slice(2, 10)}`),
      tenantId: tenant,
      ownerId: user,
      collectionId: coll,
      documentId: doc,
      modality: ChunkModality.Text,
      text,
      embedding: v,
      metadata: { company: 'acme' },
      tokenCount: text.split(' ').length,
      createdAt: new Date(),
    });
    await store.add(c);
  }
};

const integration = process.env['RAGHUB_RUN_SQLITE_TESTS'] === '1';
const itg = integration ? it : it.skip;

describe('SqliteVecStore + Retrieval (integration)', () => {
  let store: SqliteVecStore;
  let embedder: FeatureHashingEmbedder;

  beforeAll(async () => {
    store = new SqliteVecStore({ path: ':memory:', embeddingDim: 128 });
    embedder = new FeatureHashingEmbedder('fh', 128);
    await seed(store, embedder);
  }, 30_000);

  afterAll(async () => {
    await store.close();
  });

  itg('vector search returns the most semantically similar chunk', async () => {
    const u = mkUser('admin');
    const r = new Retrieval(embedder, store, { topK: 2 });
    const hits = await r.retrieve(u, 'what is raghub?');
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]?.chunk.text.toLowerCase()).toContain('raghub');
  });

  itg('keyword search returns BM25-ranked hits', async () => {
    const hits = await store.searchKeyword({
      query: 'raghub framework',
      topK: 3,
      filter: { tenantId: tenant, userId: null, collectionId: null, allowedCompanies: [] },
    });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]?.text.toLowerCase()).toContain('raghub');
  });

  itg('vector search respects tenant isolation', async () => {
    const otherTenant = brandId<TenantId>('tnt_2');
    const hits = await store.searchVector({
      vector: new Array(128).fill(0),
      topK: 10,
      filter: {
        tenantId: otherTenant,
        userId: null,
        collectionId: null,
        allowedCompanies: [],
      },
    });
    expect(hits).toEqual([]);
  });
});