/**
 * Branded identifier types.
 *
 * `string`-shaped at runtime (zero-cost) so they pass through JSON
 * unchanged, but nominal at compile time so a `UserId` cannot be
 * silently passed where a `WorkspaceId` is expected.
 */

declare const brand: unique symbol;

export type Brand<TBase, TName extends string> = TBase & {
  readonly [brand]: TName;
};

export type WorkspaceId = Brand<string, 'WorkspaceId'>;
export type UserId = Brand<string, 'UserId'>;
export type CollectionId = Brand<string, 'CollectionId'>;
export type DocumentId = Brand<string, 'DocumentId'>;
export type ChunkId = Brand<string, 'ChunkId'>;
export type SessionId = Brand<string, 'SessionId'>;
export type TraceId = Brand<string, 'TraceId'>;
export type JobId = Brand<string, 'JobId'>;

/**
 * Mint a branded ID from an unbranded string.
 *
 * Use only at the trust boundary (database row reads, JWT claims,
 * user input). Domain code should treat branded IDs as opaque.
 */
export const brandId = <T extends Brand<string, string>>(raw: string): T => raw as T;