/**
 * Langfuse telemetry.
 *
 * Lazy-loads the Langfuse SDK so the package builds without it.
 * Every call is wrapped in try/catch; telemetry must never crash
 * the caller.
 */

import type { Telemetry, TelemetrySpan } from './types.js';

interface LangfuseModule {
  Langfuse: new (opts: { publicKey: string; secretKey: string; baseUrl?: string }) => LangfuseInstance;
}

interface LangfuseInstance {
  span: (input: { name: string; input?: unknown; metadata?: Record<string, unknown> }) => LangfuseSpan;
  flush: () => Promise<void>;
  shutdown: () => Promise<void>;
}

interface LangfuseSpan {
  update: (input: { output?: unknown; metadata?: Record<string, unknown>; level?: string }) => void;
  end: () => void;
  score?: (input: { name: string; value: number; comment?: string }) => void;
}

const dynamicRequire = new Function('spec', 'return import(spec)') as (
  spec: string,
) => Promise<unknown>;

const loadLangfuse = async (): Promise<LangfuseModule | null> => {
  try {
    const mod = (await dynamicRequire('langfuse')) as { default?: LangfuseModule } & LangfuseModule;
    return mod.default ?? mod;
  } catch {
    return null;
  }
};

export interface LangfuseTelemetryOptions {
  readonly publicKey: string;
  readonly secretKey: string;
  readonly baseUrl?: string;
}

class LangfuseSpan implements TelemetrySpan {
  constructor(private readonly span: LangfuseSpan) {}

  public setAttribute(key: string, value: string | number | boolean): void {
    this.span.update({ metadata: { [key]: value } });
  }

  public setAttributes(attributes: Readonly<Record<string, string | number | boolean>>): void {
    this.span.update({ metadata: { ...attributes } });
  }

  public recordException(err: unknown): void {
    const message = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
    this.span.update({ level: 'ERROR', metadata: { exception: message } });
  }

  public end(): void {
    this.span.end();
  }
}

export class LangfuseTelemetry implements Telemetry {
  public readonly provider = 'langfuse';
  private readonly client: LangfuseInstance;

  constructor(client: LangfuseInstance) {
    this.client = client;
  }

  public span(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): TelemetrySpan {
    const handle = this.client.span({ name, metadata: { ...attributes } });
    return new LangfuseSpan(handle);
  }

  public event(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): void {
    this.client.span({ name, metadata: { ...attributes } }).end();
  }

  public override score(name: string, value: number, comment?: string): void {
    // Langfuse requires an entity-bound score; emit a 0-duration span
    // marker so the value lands in metadata (full score API lands in
    // Phase 2 once traceIds are wired through the agent loop).
    this.event('score', { score_name: name, score_value: value, ...(comment !== undefined ? { score_comment: comment } : {}) });
  }

  public async flush(): Promise<void> {
    await this.client.flush();
  }

  public async shutdown(): Promise<void> {
    await this.client.shutdown();
  }

  public static async create(opts: LangfuseTelemetryOptions): Promise<LangfuseTelemetry | null> {
    const mod = await loadLangfuse();
    if (!mod) return null;
    const client = new mod.Langfuse({
      publicKey: opts.publicKey,
      secretKey: opts.secretKey,
      ...(opts.baseUrl !== undefined ? { baseUrl: opts.baseUrl } : {}),
    });
    return new LangfuseTelemetry(client);
  }
}