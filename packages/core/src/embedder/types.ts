/**
 * Embedding provider contract.
 *
 * One concrete impl per provider (openai, litellm, feature_hashing).
 * `embedQuery` is for the user question; `embedDocuments` is for the
 * ingest path and may batch.
 *
 * Models must produce the same vector space across both methods;
 * that is the caller's responsibility (the embedder does not enforce
 * it). Mismatched dimensions throw `ConfigurationError` at startup.
 */

export interface Embedder {
  /** Stable model identifier used in settings + telemetry. */
  readonly model: string;

  /** Vector dimension returned by `embed*`. */
  readonly dimension: number;

  embedQuery(text: string): Promise<readonly number[]>;

  embedDocuments(texts: readonly string[]): Promise<readonly (readonly number[])[]>;
}