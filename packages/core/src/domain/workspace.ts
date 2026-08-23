/**
 * Workspace — the top-level multi-tenant boundary.
 *
 * A tenant owns its own users, collections, documents, and (eventually)
 * its own trace corpus. The active implementation uses RowLevel
 * isolation only, so a tenant is a logical row in the `tenants` table
 * with a stable `workspace_id` that every store query filters on.
 */

import type { WorkspaceId } from './ids.js';

export const WorkspacePlan = {
  Free: 'free',
  Pro: 'pro',
  Enterprise: 'enterprise',
} as const;

export type WorkspacePlanValue = (typeof WorkspacePlan)[keyof typeof WorkspacePlan];

export interface WorkspaceProps {
  readonly id: WorkspaceId;
  readonly name: string;
  readonly plan: WorkspacePlanValue;
  readonly createdAt: Date;
  readonly isAdmin: boolean;
}

/**
 * Frozen value object. Construct via the factory; clone via
 * `with*` helpers (none yet — add only when mutation becomes a
 * real requirement).
 */
export class Workspace {
  private readonly props: WorkspaceProps;

  constructor(props: WorkspaceProps) {
    this.props = Object.freeze({ ...props });
  }

  public get id(): WorkspaceId {
    return this.props.id;
  }

  public get name(): string {
    return this.props.name;
  }

  public get plan(): WorkspacePlanValue {
    return this.props.plan;
  }

  public get createdAt(): Date {
    return this.props.createdAt;
  }

  public get isAdmin(): boolean {
    return this.props.isAdmin;
  }

  public toJSON(): WorkspaceProps {
    return { ...this.props, createdAt: this.props.createdAt };
  }
}