/**
 * Build the `invocation_state` from a request context.
 *
 * Single read site for the tenant + user + session tuple. Tools and
 * hooks read this through the Strands `context.invocationState`
 * accessor; the orchestrator stores it under `__raghub_invocation`
 * on its `MultiAgentState` so it survives node-to-node hand-offs.
 */

import type {
  CollectionId,
  TenantId,
  UserId,
  User,
} from '@raghub/core';
import { allowedCompanyFilter, brandId } from '@raghub/core';

import type { InvocationState, Strategy } from './types.js';

export interface BuildInvocationStateInput {
  readonly tenantId: TenantId;
  readonly user: User | null;
  readonly sessionId: string | null;
  readonly sessionOverrides?: Readonly<Record<string, unknown>>;
  readonly strategy: Strategy;
  readonly traceId?: string | null;
  readonly requestId?: string | null;
  readonly db?: unknown;
  readonly secrets?: unknown;
}

export const buildInvocationState = (input: BuildInvocationStateInput): InvocationState => {
  const user = input.user;
  const rbacFilter = user
    ? allowedCompanyFilter(user)
    : {
        tenantId: input.tenantId,
        userId: null as UserId | null,
        collectionId: null as CollectionId | null,
        allowedCompanies: [] as readonly string[],
      };
  const collectionId: CollectionId | null = rbacFilter.collectionId
    ? brandId<CollectionId>(rbacFilter.collectionId)
    : null;
  return Object.freeze({
    tenant_id: input.tenantId,
    user_id: user?.id ?? null,
    is_admin: user?.isAdmin ?? false,
    rbac_filter: Object.freeze({
      tenantId: rbacFilter.tenantId,
      userId: rbacFilter.userId,
      collectionId,
      allowedCompanies: rbacFilter.allowedCompanies,
    }),
    session_id: input.sessionId,
    session_overrides: Object.freeze({ ...(input.sessionOverrides ?? {}) }),
    strategy: input.strategy,
    trace_id: input.traceId ?? null,
    request_id: input.requestId ?? null,
    db: input.db ?? null,
    secrets: input.secrets ?? null,
  });
};