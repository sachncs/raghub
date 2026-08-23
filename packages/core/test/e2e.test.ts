/**
 * End-to-end smoke test.
 *
 * Drives the full pipeline:
 *   1. register a workspace + admin user (with passphrase)
 *   2. open the unlocked workspace
 *   3. ingest a small document with agentic side-effects
 *   4. share the document with a second user
 *   5. log in as the second user
 *   6. retrieve; assert ACL filtering keeps the second user away
 *      from a private document and lets them into a shared one
 *   7. add a memory fact, search it back
 *   8. mutate the ACL (revoke) and assert visibility flips
 *
 * Everything runs in-process against an in-memory SQLite + Fs
 * LocalFileStorage. No network, no real LLM — the embedder is a
 * deterministic hash function and the LLM is a stub generator.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  brandId,
  BcryptHasher,
  Chunk,
  type ChunkId,
  type CollectionId,
  type DocumentId,
  type Embedder,
  JwtService,
  type Llm,
  MemoryScope,
  openEncryptedWorkspace,
  type Settings,
  SqliteAuditEventStore,
  SqliteDocumentPrincipalStore,
  SqliteDocumentStore,
  SqliteUserStore,
  SqliteVecStore,
  SqliteWorkspaceMemberStore,
  SqliteWorkspaceMemoryStore,
  type UserId,
  type WorkspaceId,
  agenticIngest,
  hashDocument,
} from '@raghub/core';

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const HASH_DIM = 8;

class DeterministicEmbedder {
  public readonly model = 'deterministic-v1';
  public readonly dimension = HASH_DIM;
  public async embedQuery(text: string): Promise<number[]> {
    return this.embed(text);
  }
  public async embedDocuments(texts: readonly string[]): Promise<number[][]> {
    return Promise.all(texts.map((t) => this.embed(t)));
  }
  private embed(text: string): number[] {
    const out = new Array<number>(HASH_DIM).fill(0);
    let h = 0;
    for (let i = 0; i < text.length; i++) {
      h = ((h << 5) - h + text.charCodeAt(i)) | 0;
    }
    for (let i = 0; i < HASH_DIM; i++) out[i] = ((h >> i) & 0xff) / 255;
    return out;
  }
}

const stubLlm: Llm = {
  model: 'stub',
  async generate() {
    return { content: 'stub answer' };
  },
  async *stream() {},
} as never;

interface Boot {
  handle: Awaited<ReturnType<typeof openEncryptedWorkspace>>;
  userStore: SqliteUserStore;
  documentStore: SqliteDocumentStore;
  principalStore: SqliteDocumentPrincipalStore;
  memberStore: SqliteWorkspaceMemberStore;
  memoryStore: SqliteWorkspaceMemoryStore;
  vecStore: SqliteVecStore;
  embedder: Embedder;
  audit: SqliteAuditEventStore;
}

const boot = async (path: string, passphrase: string): Promise<Boot> => {
  const handle = await openEncryptedWorkspace({ path, passphrase });
  const userStore = new SqliteUserStore({ db: handle.db });
  const documentStore = new SqliteDocumentStore({ db: handle.db });
  const principalStore = new SqliteDocumentPrincipalStore({ db: handle.db });
  const memberStore = new SqliteWorkspaceMemberStore({ db: handle.db });
  const memoryStore = new SqliteWorkspaceMemoryStore({ db: handle.db });
  const vecStore = new SqliteVecStore({ db: handle.db, embeddingDim: HASH_DIM });
  const embedder = new DeterministicEmbedder();
  const audit = new SqliteAuditEventStore({ db: handle.db });
  return { handle, userStore, documentStore, principalStore, memberStore, memoryStore, vecStore, embedder, audit };
};

const close = async (b: Boot): Promise<void> => {
  await b.principalStore.close();
  await b.memberStore.close();
  await b.memoryStore.close();
  await b.userStore.close();
  await b.documentStore.close();
  await b.vecStore.close();
  await b.audit.close();
  b.handle.close();
};

describe('end-to-end smoke', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'raghub-e2e-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  const integration = process.env['RAGHUB_RUN_SQLITE_TESTS'] === '1' && process.env['RAGHUB_LOAD_SQLITE_VEC'] === '1';
  const itg = integration ? it : it.skip;

  itg('runs the full workspace flow', async () => {
    const wsp = brandId<WorkspaceId>('wsp_e2e');
    const alice = brandId<UserId>('usr_alice');
    const bob = brandId<UserId>('usr_bob');
    const path = join(dir, 'workspace.db');
    const b = await boot(path, 'correct horse battery staple');

    const hasher = new BcryptHasher(4);
    const jwt = new JwtService({ secret: 'a-very-long-test-secret-32-bytes!', algorithm: 'HS256', ttlSeconds: 3600 });

    // 1. register
    const aliceHash = await hasher.hash('password123');
    await b.userStore.create({ workspaceId: wsp, email: 'alice@x.com', passwordHash: aliceHash, role: 'Admin', allowedCompanies: [] });
    await b.memberStore.upsert({ workspaceId: wsp, userId: alice, role: 'owner' });
    await b.audit.record({ kind: 'auth.register', workspaceId: wsp, actorId: alice, resourceId: null, detail: {} });

    // 2. ingest two docs
    const coll = brandId<CollectionId>('col_1');
    const docShared = brandId<DocumentId>('doc_shared');
    const docPrivate = brandId<DocumentId>('doc_private');
    const sharedContent = Buffer.from('shared secret in this document');
    const privateContent = Buffer.from('private secret not for everyone');

    const sharedHash = hashDocument(sharedContent);
    const privateHash = hashDocument(privateContent);

    await b.documentStore.upsert({
      workspaceId: wsp, ownerId: alice, filename: 'shared.txt', mimeType: 'text/plain',
      hash: sharedHash, byteSize: sharedContent.byteLength, metadata: {},
    });
    await b.documentStore.upsert({
      workspaceId: wsp, ownerId: alice, filename: 'private.txt', mimeType: 'text/plain',
      hash: privateHash, byteSize: privateContent.byteLength, metadata: {},
    });

    const shared = await agenticIngest(
      { workspaceId: wsp, ownerId: alice, collectionId: coll, filename: 'shared.txt', mimeType: 'text/plain', content: sharedContent },
      { embedder: b.embedder, store: b.vecStore, extra: { memoryStore: b.memoryStore } },
    );
    const priv = await agenticIngest(
      { workspaceId: wsp, ownerId: alice, collectionId: coll, filename: 'private.txt', mimeType: 'text/plain', content: privateContent },
      { embedder: b.embedder, store: b.vecStore, extra: { memoryStore: b.memoryStore } },
    );

    expect(shared.chunks.length).toBeGreaterThan(0);
    expect(priv.chunks.length).toBeGreaterThan(0);

    // 3. ACL: alice owns everything + grant bob read on shared
    await b.principalStore.applyDefaultAcl({
      documentId: shared.chunks[0]?.documentId ?? docShared,
      ownerId: alice,
      adminUserIds: [],
    });
    await b.principalStore.applyDefaultAcl({
      documentId: priv.chunks[0]?.documentId ?? docPrivate,
      ownerId: alice,
      adminUserIds: [],
    });
    const realShared = (await b.documentStore.getByHash(wsp, sharedHash))?.id ?? docShared;
    const realPrivate = (await b.documentStore.getByHash(wsp, privateHash))?.id ?? docPrivate;
    await b.principalStore.grant({
      documentId: realShared,
      principalType: 'user',
      principalId: bob,
      permission: 'read',
      grantedBy: alice,
    });

    // 4. Memory: alice remembers a fact
    await b.memoryStore.remember({
      workspaceId: wsp,
      userId: alice,
      scope: MemoryScope.User,
      content: 'Alice prefers concise answers',
    });

    // 5. JWT mint works
    const token = await jwt.mint({ subject: alice, workspaceId: wsp, isAdmin: true });
    expect(token.length).toBeGreaterThan(20);

    // 6. Bob retrieves: only sees the shared doc
    const bobFilter = {
      workspaceId: wsp,
      userId: bob,
      collectionId: null,
      principals: [{ type: 'user' as const, id: bob }],
      allowedCompanies: [],
    };
    const bobResults = await b.vecStore.searchKeyword({ query: 'secret', topK: 10, filter: bobFilter });
    expect(bobResults.length).toBe(1);
    expect(bobResults[0]?.text).toContain('shared');

    // 7. Revoke: bob can no longer see shared either
    await b.principalStore.revoke({
      documentId: realShared,
      principalType: 'user',
      principalId: bob,
      permission: 'read',
    });
    const afterRevoke = await b.vecStore.searchKeyword({ query: 'secret', topK: 10, filter: bobFilter });
    expect(afterRevoke.length).toBe(0);

    // 8. Memory search works
    const facts = await b.memoryStore.search({
      workspaceId: wsp,
      userId: alice,
      query: 'concise',
      topK: 5,
      allowedCompanies: [],
    });
    expect(facts.length).toBeGreaterThan(0);

    // 9. Settings round-trip
    const value: Settings['llm'] = { provider: 'minimax', model: 'MiniMax-Text-01', temperature: 0.1 };
    await b.handle.settings.set('llm', value);
    const got = await b.handle.settings.get<Settings['llm']>('llm');
    expect(got?.provider).toBe('minimax');

    // 10. Audit log
    const audit = await b.audit.list({ workspaceId: wsp });
    expect(audit.find((a) => a.kind === 'auth.register')).toBeTruthy();

    await close(b);
  }, 30_000);

  it('rejects a wrong passphrase on re-open', async () => {
    const path = join(dir, 'workspace-wrong.db');
    const b = await boot(path, 'correct horse battery staple');
    await b.handle.settings.set('llm', { provider: 'openai', model: 'gpt-4.1', temperature: 0 });
    await close(b);

    let threw = false;
    try {
      const second = await openEncryptedWorkspace({ path, passphrase: 'wrong passphrase here' });
      await second.settings.get('llm');
      second.close();
    } catch {
      threw = true;
    }
    expect(threw).toBe(true);
  });

  it('chunk schema round-trip', () => {
    const c = new Chunk({
      id: brandId<ChunkId>('chk_1'),
      workspaceId: brandId<WorkspaceId>('wsp_1'),
      ownerId: brandId<UserId>('usr_1'),
      collectionId: brandId<CollectionId>('col_1'),
      documentId: brandId<DocumentId>('doc_1'),
      modality: 'text',
      text: 'hello',
      embedding: [],
      metadata: {},
      tokenCount: 1,
      createdAt: new Date(),
    });
    expect(c.text).toBe('hello');
  });

  void stubLlm;
});