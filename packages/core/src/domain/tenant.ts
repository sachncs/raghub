/**
 * Tenant — the top-level multi-tenant boundary.
 *
 * A tenant owns its own users, collections, documents, and (eventually)
 * its own trace corpus. The active implementation uses RowLevel
 * isolation only, so a tenant is a logical row in the `tenants` table
 * with a stable `tenant_id` that every store query filters on.
 */

import type { TenantId } from './ids.js';

export const TenantPlan = {
  Free: 'free',
  Pro: 'pro',
  Enterprise: 'enterprise',
} as const;

export type TenantPlanValue = (typeof TenantPlan)[keyof typeof TenantPlan];

export interface TenantProps {
  readonly id: TenantId;
  readonly name: string;
  readonly plan: TenantPlanValue;
  readonly createdAt: Date;
  readonly isAdmin: boolean;
}

/**
 * Frozen value object. Construct via the factory; clone via
 * `with*` helpers (none yet — add only when mutation becomes a
 * real requirement).
 */
export class Tenant {
  private readonly props: TenantProps;

  constructor(props: TenantProps) {
    this.props = Object.freeze({ ...props });
  }

  public get id(): TenantId {
    return this.props.id;
  }

  public get name(): string {
    return this.props.name;
  }

  public get plan(): TenantPlanValue {
    return this.props.plan;
  }

  public get createdAt(): Date {
    return this.props.createdAt;
  }

  public get isAdmin(): boolean {
    return this.props.isAdmin;
  }

  public toJSON(): TenantProps {
    return { ...this.props, createdAt: this.props.createdAt };
  }
}