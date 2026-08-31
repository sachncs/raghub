/**
 * Token estimation.
 *
 * Heuristic: 1 token ≈ 4 characters for English prose (OpenAI's
 * published rule of thumb). This avoids pulling in a real tokenizer
 * (and a 5MB WASM blob) into the hot path of every LLM call and
 * embedder batch. The actual token count from the provider's
 * `usage` field is preferred when available — this estimator is
 * the fallback for chunk sizing, context budgeting, and pre-flight
 * checks.
 *
 * Calibrate against your corpus if precision matters: e.g.
 * `gpt-tokenizer` reports ~0.85× the heuristic on English code.
 */

export const estimateTokens = (text: string): number =>
  Math.max(1, Math.ceil(text.length / 4));

export const estimateMessagesTokens = (
  messages: ReadonlyArray<{ role: string; content: string }>,
): number =>
  messages.reduce((sum, m) => sum + estimateTokens(m.content) + 4, 0);