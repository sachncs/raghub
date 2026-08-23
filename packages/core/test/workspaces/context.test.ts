import { describe, expect, it } from 'vitest';

import { brandId } from '../../src/domain/index.js';
import type { WorkspaceId, UserId } from '../../src/domain/index.js';
import { AuthorizationError } from '../../src/errors/index.js';
import {
  currentWorkspace,
  requireWorkspace,
  runWithWorkspaceAsync,
  tenantContext,
} from '../../src/workspaces/index.js';

const workspaceId = brandId<WorkspaceId>('tnt_1');
const userId = brandId<UserId>('usr_1');

const ctx = tenantContext({ workspaceId, userId, isAdmin: true });

describe('tenant context', () => {
  it('propagates the context through async boundaries', async () => {
    const result = await runWithWorkspaceAsync(ctx, async () => {
      const got = currentWorkspace();
      await Promise.resolve();
      return got;
    });
    expect(result).toEqual(ctx);
  });

  it('requireWorkspace throws when no context is bound', () => {
    expect(() => requireWorkspace()).toThrow(AuthorizationError);
  });

  it('admin context exposes isAdmin', () => {
    runWithWorkspaceAsync(ctx, async () => {
      expect(requireWorkspace().isAdmin).toBe(true);
      expect(requireWorkspace().workspaceId).toBe(workspaceId);
    });
  });

  it('anonymous context has null userId', () => {
    const anon = tenantContext({ workspaceId, userId: null, isAdmin: false });
    runWithWorkspaceAsync(anon, async () => {
      expect(requireWorkspace().userId).toBeNull();
    });
  });
});