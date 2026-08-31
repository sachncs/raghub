import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { brandId, UserRole, type WorkspaceId, type UserId } from '../../src/domain/index.js';
import { BcryptHasher } from '../../src/auth/password.js';
import { openWorkspace, type WorkspaceHandle } from '../../src/workspace.js';
import { SqliteUserStore } from '../../src/storage/users.js';

const workspaceId = brandId<WorkspaceId>('wsp_1');
const otherTenantId = brandId<WorkspaceId>('wsp_2');

const setupStore = async (): Promise<{ workspace: WorkspaceHandle; store: SqliteUserStore }> => {
  const workspace = await openWorkspace({ path: ':memory:' });
  const store = new SqliteUserStore({ db: workspace.db });
  return { workspace, store };
};

const integration = process.env['REVEX_RUN_SQLITE_TESTS'] === '1';
const itg = integration ? it : it.skip;

describe('SqliteUserStore (integration)', () => {
  let store: SqliteUserStore;
  let workspace: WorkspaceHandle;

  beforeEach(async () => {
    const s = await setupStore();
    workspace = s.workspace;
    store = s.store;
  });

  afterEach(async () => {
    await store.close();
    workspace.close();
  });

  itg('creates a user, hashes the password, and looks it up by email', async () => {
    const hasher = new BcryptHasher(4);
    const hash = await hasher.hash('hunter2');
    const user = await store.create({
      workspaceId,
      email: 'a@x',
      passwordHash: hash,
      role: 'Member',
      allowedCompanies: ['acme'],
    });
    expect(user.email).toBe('a@x');
    expect(user.role).toBe(UserRole.Member);
    const lookup = await store.getByEmail('a@x');
    expect(lookup?.passwordHash).toBe(hash);
    expect(await hasher.verify('hunter2', lookup?.passwordHash ?? '')).toBe(true);
    expect(await hasher.verify('wrong', lookup?.passwordHash ?? '')).toBe(false);
  });

  itg('looks up a user by id scoped to the workspace', async () => {
    const hasher = new BcryptHasher(4);
    const user = await store.create({
      workspaceId,
      email: 'a@x',
      passwordHash: await hasher.hash('p'),
      role: 'Member',
      allowedCompanies: [],
    });
    const ok = await store.getById(workspaceId, user.id);
    expect(ok?.id).toBe(user.id);
    const cross = await store.getById(otherTenantId, user.id);
    expect(cross).toBeNull();
  });
});


