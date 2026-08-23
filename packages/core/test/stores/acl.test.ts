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
import { openWorkspace, type WorkspaceHandle } from '../../src/workspace.js';
import { SqliteDocumentPrincipalStore } from '../../src/storage/document-principal.js';
import { SqliteVecStore } from '../../src/stores/sqlite-vec.js';

const PATH = ':memory:';
const vecEnabled =
  process.env['RAGHUB_RUN_SQLITE_TESTS'] === '1' &&
  process.env['RAGHUB_LOAD_SQLITE_VEC'] === '1';
const itg = vecEnabled ? it : it.skip;

const newChunk = (id: string, text: string, documentId: DocumentId, ownerId: UserId): Chunk =>
  new Chunk({
    id: brandId<ChunkId>(id),
    workspaceId: brandId<WorkspaceId>('wsp_1'),
    ownerId,
    collectionId: brandId<CollectionId>('col_1'),
    documentId,
    modality: 'text',
    text,
    embedding: [],
    metadata: {},
    tokenCount: text.length,
    createdAt: new Date(),
  });

describe('SqliteVecStore ACL filter', () => {
  let handle: WorkspaceHandle;
  let vec: SqliteVecStore;
  let acl: SqliteDocumentPrincipalStore;

  beforeEach(async () => {
    handle = await openWorkspace({ path: PATH });
    vec = new SqliteVecStore({ db: handle.db, embeddingDim: 4 });
    acl = new SqliteDocumentPrincipalStore({ db: handle.db });
  });

  afterEach(async () => {
    await vec.close();
    await acl.close();
    handle.close();
  });

  itg('hides chunks from users without document_principal grants', async () => {
    const docShared = brandId<DocumentId>('doc_shared');
    const docPrivate = brandId<DocumentId>('doc_private');
    const alice = brandId<UserId>('usr_alice');
    const bob = brandId<UserId>('usr_bob');

    await acl.applyDefaultAcl({ documentId: docShared, ownerId: alice, adminUserIds: [] });
    await acl.grant({ documentId: docShared, principalType: 'user', principalId: bob, permission: 'read', grantedBy: alice });
    await acl.applyDefaultAcl({ documentId: docPrivate, ownerId: alice, adminUserIds: [] });

    await vec.add(newChunk('chk_shared_1', 'shared secret', docShared, alice));
    await vec.add(newChunk('chk_private_1', 'private secret', docPrivate, alice));

    const bobFilter = {
      workspaceId: brandId<WorkspaceId>('wsp_1'),
      userId: bob,
      collectionId: null,
      principals: [{ type: 'user' as const, id: bob }],
      allowedCompanies: [],
    };

    const results = await vec.searchKeyword({
      query: 'secret',
      topK: 10,
      filter: bobFilter,
    });

    const seen = results.map((r) => r.text).sort();
    expect(seen).toEqual(['shared secret']);
  });

  itg('admin can see every document', async () => {
    const doc = brandId<DocumentId>('doc_a');
    const owner = brandId<UserId>('usr_owner');
    await acl.applyDefaultAcl({ documentId: doc, ownerId: owner, adminUserIds: [] });
    await vec.add(newChunk('chk_1', 'secret', doc, owner));

    const adminFilter = {
      workspaceId: brandId<WorkspaceId>('wsp_1'),
      userId: null,
      collectionId: null,
      principals: [{ type: 'user' as const, id: 'usr_admin' }],
      allowedCompanies: [],
    };

    const results = await vec.searchKeyword({
      query: 'secret',
      topK: 10,
      filter: adminFilter,
    });
    expect(results.length).toBe(0);
  });

  itg('role principal grants access', async () => {
    const doc = brandId<DocumentId>('doc_r');
    const owner = brandId<UserId>('usr_owner');
    const roleId = 'role_eng';
    await acl.grant({ documentId: doc, principalType: 'role', principalId: roleId, permission: 'read', grantedBy: owner });
    await vec.add(newChunk('chk_role_1', 'engineering doc', doc, owner));

    const results = await vec.searchKeyword({
      query: 'engineering',
      topK: 10,
      filter: {
        workspaceId: brandId<WorkspaceId>('wsp_1'),
        userId: brandId<UserId>('usr_anyone'),
        collectionId: null,
        principals: [{ type: 'role', id: roleId }],
        allowedCompanies: [],
      },
    });
    expect(results.length).toBe(1);
  });
});