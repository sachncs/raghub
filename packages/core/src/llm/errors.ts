/**
 * LLM error taxonomy.
 *
 * The manager classifies every failure into a category so it can
 * pick the right retry / fallback strategy. Network errors and
 * rate limits are transient and worth retrying; auth and
 * configuration errors are terminal and surface immediately.
 */

export type LlmErrorKind =
  | 'rate_limit'
  | 'timeout'
  | 'network'
  | 'auth'
  | 'config'
  | 'context_overflow'
  | 'provider_5xx'
  | 'unknown';

export class LlmError extends Error {
  public readonly kind: LlmErrorKind;
  public readonly provider: string;
  public readonly status?: number;
  public readonly retryable: boolean;

  constructor(opts: {
    kind: LlmErrorKind;
    provider: string;
    message: string;
    status?: number;
    retryable: boolean;
    cause?: unknown;
  }) {
    super(opts.message, opts.cause ? { cause: opts.cause } : undefined);
    this.name = 'LlmError';
    this.kind = opts.kind;
    this.provider = opts.provider;
    this.retryable = opts.retryable;
    Object.setPrototypeOf(this, LlmError.prototype);
    if (opts.status !== undefined) this.status = opts.status;
  }
}

const classifyStatus = (status: number): { kind: LlmErrorKind; retryable: boolean } => {
  if (status === 401 || status === 403) return { kind: 'auth', retryable: false };
  if (status === 408) return { kind: 'timeout', retryable: true };
  if (status === 413) return { kind: 'context_overflow', retryable: false };
  if (status === 429) return { kind: 'rate_limit', retryable: true };
  if (status >= 500 && status < 600) return { kind: 'provider_5xx', retryable: true };
  return { kind: 'unknown', retryable: false };
};

const classifyMessage = (
  message: string,
): { kind: LlmErrorKind; retryable: boolean } => {
  const lower = message.toLowerCase();
  if (lower.includes('rate limit') || lower.includes('too many requests'))
    return { kind: 'rate_limit', retryable: true };
  if (lower.includes('timeout') || lower.includes('timed out'))
    return { kind: 'timeout', retryable: true };
  if (
    lower.includes('econnreset') ||
    lower.includes('enotfound') ||
    lower.includes('econnrefused') ||
    lower.includes('network') ||
    lower.includes('fetch failed')
  )
    return { kind: 'network', retryable: true };
  if (
    lower.includes('api key') ||
    lower.includes('unauthorized') ||
    lower.includes('forbidden') ||
    lower.includes('authentication')
  )
    return { kind: 'auth', retryable: false };
  if (
    lower.includes('context length') ||
    lower.includes('maximum context') ||
    lower.includes('too long') ||
    lower.includes('token limit')
  )
    return { kind: 'context_overflow', retryable: false };
  if (lower.includes('config'))
    return { kind: 'config', retryable: false };
  return { kind: 'unknown', retryable: false };
};

export const classifyError = (
  provider: string,
  err: unknown,
): LlmError => {
  if (err instanceof LlmError) return err;
  const message = err instanceof Error ? err.message : String(err);
  let status: number | undefined;
  const anyErr = err as { status?: number; statusCode?: number; response?: { status?: number } };
  if (typeof err === 'object' && err !== null) {
    status =
      anyErr?.status ?? anyErr?.statusCode ?? anyErr?.response?.status;
  }
  const byStatus =
    typeof status === 'number' ? classifyStatus(status) : null;
  const byMessage = classifyMessage(message);
  const { kind, retryable } = byStatus && byStatus.kind !== 'unknown' ? byStatus : byMessage;
  return new LlmError({
    kind,
    provider,
    message,
    ...(status !== undefined ? { status } : {}),
    retryable,
    cause: err,
  });
};