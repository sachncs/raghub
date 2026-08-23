/**
 * Build a `StoreFilter` from a `User`.
 *
 * Admins (or users with no principals specified) see everything in
 * the workspace. Members and viewers have `principals` populated
 * from their user_id + group memberships + role memberships by
 * `resolveAccessScope()` in `@raghub/agents`.
 */

import type { Principal, StoreFilter } from '../stores/types.js';
import type { User } from '../domain/user.js';

export const allowedCompanyFilter = (user: User): StoreFilter => {
  return {
    workspaceId: user.workspaceId,
    userId: user.isAdmin ? null : user.id,
    collectionId: null,
    principals: user.isAdmin ? [] : [{ type: 'user', id: user.id }],
    allowedCompanies: user.allowedCompanies,
  };
};

export const buildFilter = (
  user: User,
  principals: readonly Principal[] = user.isAdmin ? [] : [{ type: 'user', id: user.id }],
): StoreFilter => ({
  workspaceId: user.workspaceId,
  userId: user.isAdmin ? null : user.id,
  collectionId: null,
  principals,
  allowedCompanies: user.allowedCompanies,
});
