import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { brandId, type UserId, type WorkspaceId } from '../../src/domain/index.js';
import { openWorkspace, type WorkspaceHandle } from '../../src/workspace.js';
import {
  WorkspaceMemberRole,
  SqliteWorkspaceMemberStore,
  canIngest,
  canManageWorkspace,
} from '../../src/storage/workspace-member.js';

const PATH = ':memory:';

describe('SqliteWorkspaceMemberStore', () => {
  let handle: WorkspaceHandle;
  let store: SqliteWorkspaceMemberStore;
  const workspaceId = brandId<WorkspaceId>('wsp_1');

  beforeEach(async () => {
    handle = await openWorkspace({ path: PATH });
    store = new SqliteWorkspaceMemberStore({ db: handle.db });
  });

  afterEach(async () => {
    await store.close();
    handle.close();
  });

  it('inserts and reads back a member', async () => {
    const u = brandId<UserId>('usr_1');
    await store.upsert({ workspaceId, userId: u, role: WorkspaceMemberRole.Member });
    const got = await store.get(workspaceId, u);
    expect(got?.role).toBe('member');
  });

  it('updates an existing member role', async () => {
    const u = brandId<UserId>('usr_1');
    await store.upsert({ workspaceId, userId: u, role: WorkspaceMemberRole.Member });
    await store.upsert({ workspaceId, userId: u, role: WorkspaceMemberRole.Admin });
    const got = await store.get(workspaceId, u);
    expect(got?.role).toBe('admin');
  });

  it('removes a member', async () => {
    const u = brandId<UserId>('usr_1');
    await store.upsert({ workspaceId, userId: u, role: WorkspaceMemberRole.Viewer });
    await store.remove(workspaceId, u);
    const got = await store.get(workspaceId, u);
    expect(got).toBeNull();
  });

  it('lists all members', async () => {
    await store.upsert({ workspaceId, userId: brandId<UserId>('u_a'), role: WorkspaceMemberRole.Owner });
    await store.upsert({ workspaceId, userId: brandId<UserId>('u_b'), role: WorkspaceMemberRole.Viewer });
    const list = await store.list(workspaceId);
    expect(list.length).toBe(2);
  });
});

describe('role predicates', () => {
  it('owner and admin can manage', () => {
    expect(canManageWorkspace('owner')).toBe(true);
    expect(canManageWorkspace('admin')).toBe(true);
    expect(canManageWorkspace('member')).toBe(false);
    expect(canManageWorkspace('viewer')).toBe(false);
  });

  it('only viewers are blocked from ingest', () => {
    expect(canIngest('viewer')).toBe(false);
    expect(canIngest('member')).toBe(true);
    expect(canIngest('admin')).toBe(true);
    expect(canIngest('owner')).toBe(true);
  });
});
