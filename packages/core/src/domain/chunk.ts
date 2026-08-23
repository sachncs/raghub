/**
 * Chunk — the unit of retrieval.
 *
 * A chunk is a slice of a document with an associated embedding.
 * Chunks are filtered by `(tenantId, ownerId, collectionId)` at
 * every store path; admins see everything.
 *
 * `modality` defaults to `"text`; the multimodal layer (Phase 3)
 * populates `"image"`, `"table"`, `"equation"`, `"layout"`.
 */

import type { ChunkId, CollectionId, DocumentId, TenantId, UserId } from './ids.js';

export const ChunkModality = {
  Text: 'text',
  Image: 'image',
  Table: 'table',
  Equation: 'equation',
  Layout: 'layout',
  Summary: 'summary',
} as const;

export type ChunkModalityValue = (typeof ChunkModality)[keyof typeof ChunkModality];

export interface ChunkProps {
  readonly id: ChunkId;
  readonly tenantId: TenantId;
  readonly ownerId: UserId;
  readonly collectionId: CollectionId;
  readonly documentId: DocumentId;
  readonly modality: ChunkModalityValue;
  readonly text: string;
  readonly embedding: readonly number[];
  readonly metadata: Readonly<Record<string, string>>;
  readonly tokenCount: number;
  readonly createdAt: Date;
}

export class Chunk {
  private readonly props: ChunkProps;

  constructor(props: ChunkProps) {
    this.props = Object.freeze({
      ...props,
      embedding: Object.freeze([...props.embedding]),
      metadata: Object.freeze({ ...props.metadata }),
    });
  }

  public get id(): ChunkId {
    return this.props.id;
  }

  public get tenantId(): TenantId {
    return this.props.tenantId;
  }

  public get ownerId(): UserId {
    return this.props.ownerId;
  }

  public get collectionId(): CollectionId {
    return this.props.collectionId;
  }

  public get documentId(): DocumentId {
    return this.props.documentId;
  }

  public get modality(): ChunkModalityValue {
    return this.props.modality;
  }

  public get text(): string {
    return this.props.text;
  }

  public get embedding(): readonly number[] {
    return this.props.embedding;
  }

  public get metadata(): Readonly<Record<string, string>> {
    return this.props.metadata;
  }

  public get tokenCount(): number {
    return this.props.tokenCount;
  }

  public get createdAt(): Date {
    return this.props.createdAt;
  }

  public toJSON(): ChunkProps {
    return {
      ...this.props,
      embedding: [...this.props.embedding],
      metadata: { ...this.props.metadata },
      createdAt: this.props.createdAt,
    };
  }
}

/**
 * A retrieval hit — a chunk plus a relevance score from the store.
 */
export interface Hit {
  readonly chunk: Chunk;
  readonly score: number;
}