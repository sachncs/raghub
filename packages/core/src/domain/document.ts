/**
 * Document — a user-uploaded source of truth.
 *
 * Documents are content-addressed by SHA-256 (`hash`). Re-ingesting
 * the same `hash` is a no-op: the existing chunks win. A document
 * carries an opaque `metadata` bag the user can use for any
 * tenant-scoped annotations (company tag, source URL, retention).
 */

import type { DocumentId, TenantId, UserId } from './ids.js';

export const DocumentLifecycleStatus = {
  Pending: 'pending',
  Indexing: 'indexing',
  Ready: 'ready',
  Failed: 'failed',
} as const;

export type DocumentLifecycleStatusValue =
  (typeof DocumentLifecycleStatus)[keyof typeof DocumentLifecycleStatus];

export interface DocumentProps {
  readonly id: DocumentId;
  readonly tenantId: TenantId;
  readonly ownerId: UserId;
  readonly filename: string;
  readonly mimeType: string;
  readonly hash: string;
  readonly byteSize: number;
  readonly status: DocumentLifecycleStatusValue;
  readonly metadata: Readonly<Record<string, string>>;
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

export class Document {
  private readonly props: DocumentProps;

  constructor(props: DocumentProps) {
    this.props = Object.freeze({
      ...props,
      metadata: Object.freeze({ ...props.metadata }),
    });
  }

  public get id(): DocumentId {
    return this.props.id;
  }

  public get tenantId(): TenantId {
    return this.props.tenantId;
  }

  public get ownerId(): UserId {
    return this.props.ownerId;
  }

  public get filename(): string {
    return this.props.filename;
  }

  public get mimeType(): string {
    return this.props.mimeType;
  }

  public get hash(): string {
    return this.props.hash;
  }

  public get byteSize(): number {
    return this.props.byteSize;
  }

  public get status(): DocumentLifecycleStatusValue {
    return this.props.status;
  }

  public get metadata(): Readonly<Record<string, string>> {
    return this.props.metadata;
  }

  public get createdAt(): Date {
    return this.props.createdAt;
  }

  public get updatedAt(): Date {
    return this.props.updatedAt;
  }

  public toJSON(): DocumentProps {
    return {
      ...this.props,
      metadata: { ...this.props.metadata },
      createdAt: this.props.createdAt,
      updatedAt: this.props.updatedAt,
    };
  }
}