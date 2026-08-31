import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { brandId, type UserId, type WorkspaceId } from '../../src/domain/index.js';
import { openWorkspace, type WorkspaceHandle } from '../../src/workspace.js';
import {
  MemoryScope,
  SqliteWorkspaceMemoryStore,
} from '../../src/storage/memory.js';

const PATH = ':memory:';

describe('SqliteWorkspaceMemoryStore', () => {
  let handle: WorkspaceHandle;
  let mem: SqliteWorkspaceMemoryStore;
  const wsp = brandId<WorkspaceId>('wsp_1');
  const alice = brandId<UserId>('usr_alice');
  const bob = brandId<UserId>('usr_bob');

  beforeEach(async () => {
    handle = await openWorkspace({ path: PATH });
    mem = new SqliteWorkspaceMemoryStore({ db: handle.db });
  });

  afterEach(async () => {
    await mem.close();
    handle.close();
  });

  it('persists a user-scoped fact', async () => {
    const fact = await mem.remember({
      workspaceId: wsp,
      userId: alice,
      scope: MemoryScope.User,
      content: 'prefers concise answers',
    });
    expect(fact.id).toBeGreaterThan(0);
    expect(fact.scope).toBe('user');
  });

  it('persists a workspace-scoped fact', async () => {
    await mem.remember({
      workspaceId: wsp,
      userId: null,
      scope: MemoryScope.Workspace,
      content: 'analyst team focused on Q3 reports',
    });
    const list = await mem.listForUser(wsp, bob);
    expect(list.length).toBe(1);
  });

  it('user-scoped facts only return to the same user', async () => {
    await mem.remember({ workspaceId: wsp, userId: alice, scope: MemoryScope.User, content: 'alice likes tea' });
    await mem.remember({ workspaceId: wsp, userId: bob, scope: MemoryScope.User, content: 'bob prefers coffee' });
    const aliceView = await mem.listForUser(wsp, alice);
    expect(aliceView.map((f) => f.content)).toContain('alice likes tea');
    expect(aliceView.map((f) => f.content)).not.toContain('bob prefers coffee');
  });

  it('search ranks by token overlap', async () => {
    await mem.remember({
      workspaceId: wsp,
      userId: alice,
      scope: MemoryScope.User,
      content: 'revex uses sqlite-vec for vector search',
    });
    await mem.remember({
      workspaceId: wsp,
      userId: alice,
      scope: MemoryScope.User,
      content: 'revex uses BM25 via FTS5 for keyword',
    });
    const results = await mem.search({
      workspaceId: wsp,
      userId: alice,
      query: 'sqlite-vec vector',
      topK: 5,
      allowedCompanies: [],
    });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]?.content).toContain('sqlite-vec');
  });

  it('delete removes a fact', async () => {
    const f = await mem.remember({
      workspaceId: wsp,
      userId: alice,
      scope: MemoryScope.User,
      content: 'tmp fact',
    });
    await mem.delete(f.id);
    const list = await mem.listForUser(wsp, alice);
    expect(list.length).toBe(0);
  });
});