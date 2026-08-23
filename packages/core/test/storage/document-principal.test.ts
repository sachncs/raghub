import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  brandId,
  Chunk,
  type ChunkId,
  type CollectionId,
  type DocumentId,
  type UserId,
  type WorkspaceId,
} from '../../src/domain/index.js';
import {
  SqliteDocumentPrincipalStore,
  SqliteDocumentPrincipalStore as _Ignored,
} from '../../src/storage/document-principal.js';
import { openWorkspace, type WorkspaceHandle } from '../../src/workspace.js';

const PATH = ':memory:';

describe('SqliteDocumentPrincipalStore', () => {
  let handle: WorkspaceHandle;
  let store: SqliteDocumentPrincipalStore;
  const doc1 = brandId<DocumentId>('doc_a');
  const doc2 = brandId<DocumentId>('doc_b');
  const alice = brandId<UserId>('usr_alice');
  const bob = brandId<UserId>('usr_bob');
  const owner = brandId<UserId>('usr_owner');
  void _Ignored;

  beforeEach(async () => {
    handle = await openWorkspace({ path: PATH });
    store = new SqliteDocumentPrincipalStore({ db: handle.db });
  });

  afterEach(async () => {
    await store.close();
    handle.close();
  });

  it('grants and lists principals on a document', async () => {
    await store.grant({ documentId: doc1, principalType: 'user', principalId: alice, permission: 'read', grantedBy: owner });
    await store.grant({ documentId: doc1, principalType: 'user', principalId: bob, permission: 'admin', grantedBy: owner });
    const list = await store.listByDocument(doc1);
    expect(list.length).toBe(2);
  });

  it('revokes a principal', async () => {
    await store.grant({ documentId: doc1, principalType: 'user', principalId: alice, permission: 'read', grantedBy: owner });
    await store.revoke({ documentId: doc1, principalType: 'user', principalId: alice, permission: 'read' });
    expect((await store.listByDocument(doc1)).length).toBe(0);
  });

  it('lists documents a principal can see', async () => {
    await store.grant({ documentId: doc1, principalType: 'user', principalId: alice, permission: 'read', grantedBy: owner });
    await store.grant({ documentId: doc2, principalType: 'user', principalId: alice, permission: 'read', grantedBy: owner });
    const list = await store.listByPrincipal('user', alice);
    expect(list.length).toBe(2);
  });

  it('hasAccess reflects grants', async () => {
    await store.grant({ documentId: doc1, principalType: 'user', principalId: alice, permission: 'read', grantedBy: owner });
    expect(
      await store.hasAccess(doc1, [{ type: 'user', id: alice }]),
    ).toBe(true);
    expect(
      await store.hasAccess(doc1, [{ type: 'user', id: bob }]),
    ).toBe(false);
  });

  it('applyDefaultAcl grants owner + admins', async () => {
    await store.applyDefaultAcl({
      documentId: doc1,
      ownerId: alice,
      adminUserIds: [bob, brandId<UserId>('usr_carol')],
    });
    const list = await store.listByDocument(doc1);
    expect(list.length).toBe(3);
    expect(
      await store.hasAccess(doc1, [
        { type: 'user', id: alice },
        { type: 'user', id: bob },
      ]),
    ).toBe(true);
  });

  it('INSERT OR IGNORE is idempotent', async () => {
    for (let i = 0; i < 3; i++) {
      await store.grant({ documentId: doc1, principalType: 'user', principalId: alice, permission: 'read', grantedBy: owner });
    }
    expect((await store.listByDocument(doc1)).length).toBe(1);
  });

  it('type-level sanity: Chunk remains constructable for ACL tests', () => {
    const c = new Chunk({
      id: brandId<ChunkId>('chk_1'),
      workspaceId: brandId<WorkspaceId>('wsp_1'),
      ownerId: alice,
      collectionId: brandId<CollectionId>('col_1'),
      documentId: doc1,
      modality: 'text',
      text: 'hello',
      embedding: [],
      metadata: {},
      tokenCount: 1,
      createdAt: new Date(),
    });
    expect(c.documentId).toBe(doc1);
  });
});