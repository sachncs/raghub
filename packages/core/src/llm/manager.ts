/**
 * LLM Manager — retry, fallback chain, usage aggregation.
 *
 * Wraps one or more `Llm` providers and exposes the same surface
 * (`generate`, `stream`) so callers can swap providers, fall back
 * on failure, and aggregate token usage across attempts.
 *
 *   manager = new LlmManager({ primary: openai, fallbacks: [anthropic, stub] });
 *   manager.primary  → OpenAI
 *   manager.fallbacks → [Anthropic, stub]
 *
 * Retry policy: exponential backoff with jitter, capped at
 * `maxAttempts`. Only `retryable` errors are retried — auth,
 * configuration, and context-overflow errors surface immediately.
 *
 * Fallback fires only after the primary's retries are exhausted,
 * so transient blips don't trigger a model swap.
 *
 * Usage is summed across all attempts and returned via
 * `manager.usage()` so the UI / audit trail can show how many
 * tokens a single user-facing request actually cost.
 */

import type { Readable } from 'node:stream';

import type {
  ChatMessage,
  GenerateOptions,
  GenerateResult,
  Llm,
  StreamChunk,
} from './types.js';
import { classifyError, LlmError } from './errors.js';

export interface LlmManagerOptions {
  readonly primary: Llm;
  readonly fallbacks?: readonly Llm[];
  readonly maxAttempts?: number;
  readonly baseDelayMs?: number;
  readonly maxDelayMs?: number;
  readonly onAttempt?: (info: LlmAttemptInfo) => void;
}

export interface LlmAttemptInfo {
  readonly provider: string;
  readonly attempt: number;
  readonly outcome: 'success' | 'retry' | 'fallback' | 'failure';
  readonly errorKind?: string;
  readonly latencyMs: number;
  readonly usage?: GenerateResult['usage'];
}

export interface LlmUsageTotals {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
  readonly attempts: number;
  readonly failures: number;
  readonly fallbacksUsed: number;
  readonly providers: readonly { readonly provider: string; readonly calls: number }[];
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

const jitter = (factor: number): number => 0.5 + Math.random() * factor;

export class LlmManager implements Llm {
  public readonly primary: Llm;
  public readonly fallbacks: readonly Llm[];
  public readonly provider: string;
  public readonly model: string;

  private readonly maxAttempts: number;
  private readonly baseDelayMs: number;
  private readonly maxDelayMs: number;
  private readonly onAttempt: ((info: LlmAttemptInfo) => void) | undefined;
  private readonly attempts: Map<string, number> = new Map();
  private totals = {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    attempts: 0,
    failures: 0,
    fallbacksUsed: 0,
  };

  constructor(opts: LlmManagerOptions) {
    this.primary = opts.primary;
    this.fallbacks = opts.fallbacks ?? [];
    this.provider = `manager:${this.primary.provider}`;
    this.model = this.primary.model;
    this.maxAttempts = Math.max(1, opts.maxAttempts ?? 3);
    this.baseDelayMs = Math.max(50, opts.baseDelayMs ?? 250);
    this.maxDelayMs = Math.max(this.baseDelayMs, opts.maxDelayMs ?? 4_000);
    this.onAttempt = opts.onAttempt;
  }

  public usage(): LlmUsageTotals {
    const providers = Array.from(this.attempts.entries())
      .map(([provider, calls]) => ({ provider, calls }))
      .sort((a, b) => b.calls - a.calls);
    return {
      promptTokens: this.totals.promptTokens,
      completionTokens: this.totals.completionTokens,
      totalTokens: this.totals.totalTokens,
      attempts: this.totals.attempts,
      failures: this.totals.failures,
      fallbacksUsed: this.totals.fallbacksUsed,
      providers,
    };
  }

  public async generate(opts: GenerateOptions): Promise<GenerateResult> {
    const chain: Llm[] = [this.primary, ...this.fallbacks];
    let lastError: unknown = null;
    for (const llm of chain) {
      try {
        const result = await this.tryWithRetries(llm, opts);
        return result;
      } catch (err) {
        lastError = err;
        this.recordAttempt(llm.provider);
        const llmErr = err instanceof LlmError ? err : classifyError(llm.provider, err);
        if (!llmErr.retryable && llm === this.primary) {
          throw llmErr;
        }
        if (llm === this.primary) {
          this.totals.fallbacksUsed += 1;
          if (this.onAttempt) {
            this.onAttempt({
              provider: llm.provider,
              attempt: this.maxAttempts,
              outcome: 'fallback',
              errorKind: llmErr.kind,
              latencyMs: 0,
            });
          }
        }
      }
    }
    throw classifyError(this.provider, lastError);
  }

  public async *stream(opts: GenerateOptions): AsyncIterable<StreamChunk> {
    const chain: Llm[] = [this.primary, ...this.fallbacks];
    let lastError: unknown = null;
    for (const llm of chain) {
      try {
        for await (const chunk of this.tryStreamWithRetries(llm, opts)) {
          yield chunk;
        }
        return;
      } catch (err) {
        lastError = err;
        this.recordAttempt(llm.provider);
        const llmErr = err instanceof LlmError ? err : classifyError(llm.provider, err);
        if (!llmErr.retryable && llm === this.primary) {
          throw llmErr;
        }
        if (llm === this.primary) {
          this.totals.fallbacksUsed += 1;
        }
      }
    }
    throw classifyError(this.provider, lastError);
  }

  public async rawStream(opts: GenerateOptions): Promise<Readable> {
    return this.primary.rawStream(opts);
  }

  private async tryWithRetries(
    llm: Llm,
    opts: GenerateOptions,
  ): Promise<GenerateResult> {
    let lastError: unknown = null;
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      const start = Date.now();
      try {
        const result = await llm.generate(opts);
        const latencyMs = Date.now() - start;
        this.recordSuccess(llm.provider, result.usage, latencyMs);
        if (this.onAttempt) {
          this.onAttempt({
            provider: llm.provider,
            attempt,
            outcome: 'success',
            latencyMs,
            usage: result.usage,
          });
        }
        return result;
      } catch (err) {
        const llmErr = classifyError(llm.provider, err);
        lastError = llmErr;
        this.totals.failures += 1;
        this.totals.attempts += 1;
        const latencyMs = Date.now() - start;
        if (!llmErr.retryable || attempt === this.maxAttempts) {
          if (this.onAttempt) {
            this.onAttempt({
              provider: llm.provider,
              attempt,
              outcome: 'failure',
              errorKind: llmErr.kind,
              latencyMs,
            });
          }
          throw llmErr;
        }
        const delay = Math.min(
          this.maxDelayMs,
          this.baseDelayMs * 2 ** (attempt - 1) * jitter(1),
        );
        if (this.onAttempt) {
          this.onAttempt({
            provider: llm.provider,
            attempt,
            outcome: 'retry',
            errorKind: llmErr.kind,
            latencyMs,
          });
        }
        await sleep(delay);
      }
    }
    throw lastError ?? new LlmError({
      kind: 'unknown',
      provider: llm.provider,
      message: 'retries exhausted',
      retryable: false,
    });
  }

  private async *tryStreamWithRetries(
    llm: Llm,
    opts: GenerateOptions,
  ): AsyncIterable<StreamChunk> {
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      this.totals.attempts += 1;
      const start = Date.now();
      try {
        const iterator = llm.stream(opts)[Symbol.asyncIterator]();
        while (true) {
          const next = await iterator.next();
          if (next.done) {
            const latencyMs = Date.now() - start;
            if (this.onAttempt) {
              this.onAttempt({
                provider: llm.provider,
                attempt,
                outcome: 'success',
                latencyMs,
              });
            }
            return;
          }
          yield next.value;
        }
      } catch (err) {
        const llmErr = classifyError(llm.provider, err);
        this.totals.failures += 1;
        if (!llmErr.retryable || attempt === this.maxAttempts) {
          if (this.onAttempt) {
            this.onAttempt({
              provider: llm.provider,
              attempt,
              outcome: 'failure',
              errorKind: llmErr.kind,
              latencyMs: Date.now() - start,
            });
          }
          throw llmErr;
        }
        if (this.onAttempt) {
          this.onAttempt({
            provider: llm.provider,
            attempt,
            outcome: 'retry',
            errorKind: llmErr.kind,
            latencyMs: Date.now() - start,
          });
        }
        await sleep(
          Math.min(
            this.maxDelayMs,
            this.baseDelayMs * 2 ** (attempt - 1) * jitter(1),
          ),
        );
      }
    }
  }

  private recordSuccess(
    provider: string,
    usage: GenerateResult['usage'],
    latencyMs: number,
  ): void {
    this.attempts.set(provider, (this.attempts.get(provider) ?? 0) + 1);
    this.totals.attempts += 1;
    this.totals.promptTokens += usage.promptTokens;
    this.totals.completionTokens += usage.completionTokens;
    this.totals.totalTokens += usage.totalTokens;
    if (this.onAttempt) {
      this.onAttempt({
        provider,
        attempt: 1,
        outcome: 'success',
        latencyMs,
        usage,
      });
    }
  }

  private recordAttempt(provider: string): void {
    this.attempts.set(provider, (this.attempts.get(provider) ?? 0) + 1);
  }
}

export const createManager = (opts: {
  readonly primary: Llm;
  readonly fallbacks?: readonly Llm[];
  readonly onAttempt?: (info: LlmAttemptInfo) => void;
}): LlmManager =>
  new LlmManager({
    primary: opts.primary,
    ...(opts.fallbacks ? { fallbacks: opts.fallbacks } : {}),
    ...(opts.onAttempt ? { onAttempt: opts.onAttempt } : {}),
  });

export const messagesFromConversation = (
  turns: ReadonlyArray<{ role: 'user' | 'assistant'; content: string }>,
): ChatMessage[] =>
  turns.map((t) => ({ role: t.role, content: t.content }));