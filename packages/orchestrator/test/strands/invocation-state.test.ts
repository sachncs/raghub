import { describe, expect, it } from 'vitest';

import { buildInvocationState } from '../src/strands/invocation-state.js';
import { type TenantId, type UserId, User, UserRole, brandId } from '@raghub/core';

const tenantId = brandId<TenantId>('tnt_1');
const userId = brandId<UserId>('usr_1');
const user = new User({
  id: userId,
  tenantId,
  email: 'a@x',
  role: UserRole.Member,
  allowedCompanies: ['acme'],
  createdAt: new Date(),
});

describe('buildInvocationState', () => {
  it('freezes the resulting record and exposes RBAC fields', () => {
    const state = buildInvocationState({
      tenantId,
      user,
      sessionId: 'ses_1',
      strategy: {
        mode: 'graph',
        hybrid: { denseWeight: 0.6, sparseWeight: 0.4, rrfK: 60, colbert: false },
        ordering: 'standard',
        k: 10,
        reranker: 'identity',
        multimodal: { enabled: false },
        traceCorpus: { enabled: false, representation: 'semantic' },
      },
    });
    expect(Object.isFrozen(state)).toBe(true);
    expect(state.tenant_id).toBe(tenantId);
    expect(state.user_id).toBe(userId);
    expect(state.is_admin).toBe(false);
    expect(state.session_id).toBe('ses_1');
    expect(state.rbac_filter.allowedCompanies).toEqual(['acme']);
    expect(state.rbac_filter.userId).toBe(userId);
  });

  it('handles anonymous callers with null user_id', () => {
    const state = buildInvocationState({
      tenantId,
      user: null,
      sessionId: null,
      strategy: {
        mode: 'graph',
        hybrid: { denseWeight: 0.6, sparseWeight: 0.4, rrfK: 60, colbert: false },
        ordering: 'standard',
        k: 5,
        reranker: 'identity',
        multimodal: { enabled: false },
        traceCorpus: { enabled: false, representation: 'semantic' },
      },
    });
    expect(state.user_id).toBeNull();
    expect(state.is_admin).toBe(false);
    expect(state.rbac_filter.allowedCompanies).toEqual([]);
  });
});