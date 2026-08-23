import { describe, expect, it } from 'vitest';

import {
  Chunk,
  ChunkModality,
  Document,
  DocumentLifecycleStatus,
  Workspace,
  WorkspacePlan,
  Turn,
  TurnRole,
  User,
  UserRole,
  brandId,
} from '../src/domain/index.js';
import type {
  CollectionId,
  DocumentId,
  WorkspaceId,
  UserId,
} from '../src/domain/index.js';

const workspaceId = brandId<WorkspaceId>('tnt_1');
const userId = brandId<UserId>('usr_1');
const docId = brandId<DocumentId>('doc_1');
const collectionId = brandId<CollectionId>('col_1');

describe('domain', () => {
  it('Workspace is immutable and round-trips through toJSON', () => {
    const t = new Workspace({
      id: workspaceId,
      name: 'Acme',
      plan: WorkspacePlan.Pro,
      createdAt: new Date('2025-01-01T00:00:00Z'),
      isAdmin: false,
    });
    expect(t.id).toBe(workspaceId);
    expect(t.plan).toBe('pro');
    expect(t.isAdmin).toBe(false);
    expect(Object.isFrozen((t as unknown as { props: object }).props)).toBe(true);
    const json = t.toJSON();
    expect(json.id).toBe(workspaceId);
    expect(json.createdAt).toBeInstanceOf(Date);
  });

  it('User exposes isAdmin derived from role', () => {
    const admin = new User({
      id: userId,
      workspaceId,
      email: 'a@b',
      role: UserRole.Admin,
      allowedCompanies: ['acme'],
      createdAt: new Date(),
    });
    const member = new User({
      ...admin.toJSON(),
      role: UserRole.Member,
      email: 'm@b',
    });
    expect(admin.isAdmin).toBe(true);
    expect(member.isAdmin).toBe(false);
  });

  it('Document status defaults freeze metadata', () => {
    const d = new Document({
      id: docId,
      workspaceId,
      ownerId: userId,
      filename: 'r.pdf',
      mimeType: 'application/pdf',
      hash: 'abc',
      byteSize: 100,
      status: DocumentLifecycleStatus.Pending,
      metadata: { company: 'acme' },
      createdAt: new Date(),
      updatedAt: new Date(),
    });
    expect(d.status).toBe('pending');
    expect(Object.isFrozen(d.metadata)).toBe(true);
  });

  it('Chunk freezes embedding and metadata', () => {
    const c = new Chunk({
      id: brandId('chk_1'),
      workspaceId,
      ownerId: userId,
      collectionId,
      documentId: docId,
      modality: ChunkModality.Text,
      text: 'hello',
      embedding: [0.1, 0.2, 0.3],
      metadata: { source: 'web' },
      tokenCount: 1,
      createdAt: new Date(),
    });
    expect(c.modality).toBe('text');
    expect(Object.isFrozen(c.embedding)).toBe(true);
    expect(Object.isFrozen(c.metadata)).toBe(true);
  });

  it('Turn records role and content', () => {
    const t = new Turn({
      sessionId: brandId('ses_1'),
      workspaceId,
      userId,
      role: TurnRole.User,
      content: 'hi',
      createdAt: new Date(),
    });
    expect(t.role).toBe('user');
    expect(t.content).toBe('hi');
  });
});