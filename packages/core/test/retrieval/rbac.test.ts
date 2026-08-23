import { describe, expect, it } from 'vitest';

import { User, UserRole, brandId } from '../../src/domain/index.js';
import type { WorkspaceId, UserId } from '../../src/domain/index.js';
import { allowedCompanyFilter } from '../../src/retrieval/rbac.js';

const workspaceId = brandId<WorkspaceId>('tnt_1');
const userId = brandId<UserId>('usr_1');

const makeUser = (role: keyof typeof UserRole, companies: string[]) =>
  new User({
    id: userId,
    workspaceId,
    email: 'u@x',
    role: UserRole[role],
    allowedCompanies: companies,
    createdAt: new Date(),
  });

describe('allowedCompanyFilter', () => {
  it('admin sees all users in the tenant', () => {
    const u = makeUser('Admin', []);
    const f = allowedCompanyFilter(u);
    expect(f.workspaceId).toBe(workspaceId);
    expect(f.userId).toBeNull();
  });

  it('member is scoped to their own userId', () => {
    const u = makeUser('Member', ['acme']);
    const f = allowedCompanyFilter(u);
    expect(f.userId).toBe(userId);
    expect(f.allowedCompanies).toEqual(['acme']);
  });

  it('viewer with no companies has empty allow-list', () => {
    const u = makeUser('Viewer', []);
    const f = allowedCompanyFilter(u);
    expect(f.allowedCompanies).toEqual([]);
  });
});