/**
 * Build a `StoreFilter` from a user.
 *
 * Admins see everything in their tenant. Members are scoped to
 * `metadata.company ∈ allowedCompanies`. Both are tenant-isolated.
 */

import type { User } from '../domain/user.js';
import type { StoreFilter } from '../stores/types.js';

export const allowedCompanyFilter = (user: User): StoreFilter => {
  return {
    tenantId: user.tenantId,
    userId: user.isAdmin ? null : user.id,
    collectionId: null,
    allowedCompanies: user.allowedCompanies,
  };
};