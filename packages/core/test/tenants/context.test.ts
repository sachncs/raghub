import { describe, expect, it } from 'vitest';

import { brandId } from '../../src/domain/index.js';
import type { TenantId, UserId } from '../../src/domain/index.js';
import { AuthorizationError } from '../../src/errors/index.js';
import {
  currentTenant,
  requireTenant,
  runWithTenantAsync,
  tenantContext,
} from '../../src/tenants/index.js';

const tenantId = brandId<TenantId>('tnt_1');
const userId = brandId<UserId>('usr_1');

const ctx = tenantContext({ tenantId, userId, isAdmin: true });

describe('tenant context', () => {
  it('propagates the context through async boundaries', async () => {
    const result = await runWithTenantAsync(ctx, async () => {
      const got = currentTenant();
      await Promise.resolve();
      return got;
    });
    expect(result).toEqual(ctx);
  });

  it('requireTenant throws when no context is bound', () => {
    expect(() => requireTenant()).toThrow(AuthorizationError);
  });

  it('admin context exposes isAdmin', () => {
    runWithTenantAsync(ctx, async () => {
      expect(requireTenant().isAdmin).toBe(true);
      expect(requireTenant().tenantId).toBe(tenantId);
    });
  });

  it('anonymous context has null userId', () => {
    const anon = tenantContext({ tenantId, userId: null, isAdmin: false });
    runWithTenantAsync(anon, async () => {
      expect(requireTenant().userId).toBeNull();
    });
  });
});