import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  brandId,
  TenantPlan,
  UserRole,
  type TenantId,
} from '../../src/domain/index.js';
import { BcryptHasher } from '../../src/auth/password.js';
import { SqliteUserStore } from '../../src/storage/users.js';

const tenantId = brandId<TenantId>('tnt_1');
const otherTenantId = brandId<TenantId>('tnt_2');

const setupStore = async () => {
  const store = new SqliteUserStore({ path: ':memory:' });
  await store.upsertTenant({ id: tenantId, name: 'Acme', plan: 'Free' });
  await store.upsertTenant({ id: otherTenantId, name: 'Other', plan: 'Free' });
  return store;
};

const integration = process.env['RAGHUB_RUN_SQLITE_TESTS'] === '1';
const itg = integration ? it : it.skip;

describe('SqliteUserStore (integration)', () => {
  let store: SqliteUserStore;

  beforeEach(async () => {
    store = await setupStore();
  });

  afterEach(async () => {
    await store.close();
  });

  itg('upserts a tenant and reads it back', async () => {
    const t = await store.getTenant(tenantId);
    expect(t?.name).toBe('Acme');
    expect(t?.plan).toBe(TenantPlan.Free);
  });

  itg('creates a user, hashes the password, and looks it up by email', async () => {
    const hasher = new BcryptHasher(4);
    const hash = await hasher.hash('hunter2');
    const user = await store.create({
      tenantId,
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

  itg('looks up a user by id scoped to the tenant', async () => {
    const hasher = new BcryptHasher(4);
    const user = await store.create({
      tenantId,
      email: 'a@x',
      passwordHash: await hasher.hash('p'),
      role: 'Member',
      allowedCompanies: [],
    });
    const ok = await store.getById(tenantId, user.id);
    expect(ok?.id).toBe(user.id);
    const cross = await store.getById(otherTenantId, user.id);
    expect(cross).toBeNull();
  });
});