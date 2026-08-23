import { describe, expect, it } from 'vitest';

import { User, UserRole, brandId } from '../src/domain/index.js';
import type { TenantId, UserId } from '../src/domain/index.js';
import { allowedCompanyFilter } from '../src/retrieval/rbac.js';

const tenantId = brandId<TenantId>('tnt_1');
const userId = brandId<UserId>('usr_1');

const makeUser = (role: UserRole, companies: string[]) =>
  new User({
    id: userId,
    tenantId,
    email: 'u@x',
    role,
    allowedCompanies: companies,
    createdAt: new Date(),
  });

describe('allowedCompanyFilter', () => {
  it('admin sees all users in the tenant', () => {
    const u = makeUser(UserRole.Admin, []);
    const f = allowedCompanyFilter(u);
    expect(f.tenantId).toBe(tenantId);
    expect(f.userId).toBeNull();
  });

  it('member is scoped to their own userId', () => {
    const u = makeUser(UserRole.Member, ['acme']);
    const f = allowedCompanyFilter(u);
    expect(f.userId).toBe(userId);
    expect(f.allowedCompanies).toEqual(['acme']);
  });

  it('viewer with no companies has empty allow-list', () => {
    const u = makeUser(UserRole.Viewer, []);
    const f = allowedCompanyFilter(u);
    expect(f.allowedCompanies).toEqual([]);
  });
});