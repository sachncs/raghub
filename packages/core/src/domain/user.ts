/**
 * User — a principal authenticated against a tenant.
 *
 * `isAdmin` is the only authorisation bit that escapes tenant scoping;
 * everything else is filtered by `(tenantId, userId, collectionId)`.
 */

import type { TenantId, UserId } from './ids.js';

export const UserRole = {
  Admin: 'admin',
  Member: 'member',
  Viewer: 'viewer',
} as const;

export type UserRoleValue = (typeof UserRole)[keyof typeof UserRole];

export interface UserProps {
  readonly id: UserId;
  readonly tenantId: TenantId;
  readonly email: string;
  readonly role: UserRoleValue;
  readonly allowedCompanies: readonly string[];
  readonly createdAt: Date;
}

export class User {
  private readonly props: UserProps;

  constructor(props: UserProps) {
    this.props = Object.freeze({ ...props, allowedCompanies: [...props.allowedCompanies] });
  }

  public get id(): UserId {
    return this.props.id;
  }

  public get tenantId(): TenantId {
    return this.props.tenantId;
  }

  public get email(): string {
    return this.props.email;
  }

  public get role(): UserRoleValue {
    return this.props.role;
  }

  public get allowedCompanies(): readonly string[] {
    return this.props.allowedCompanies;
  }

  public get isAdmin(): boolean {
    return this.props.role === UserRole.Admin;
  }

  public get createdAt(): Date {
    return this.props.createdAt;
  }

  public toJSON(): UserProps {
    return {
      ...this.props,
      allowedCompanies: [...this.props.allowedCompanies],
      createdAt: this.props.createdAt,
    };
  }
}