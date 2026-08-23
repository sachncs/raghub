/**
 * LLM provider contract.
 *
 * `generate()` returns the full answer when streaming is not
 * needed. `stream()` yields chunks for incremental output. Both
 * take a `ChatMessage[]` and the model name; tools/JSON-schema
 * mode is opt-in via `tools` + `responseFormat`.
 *
 * Implementations: `OpenAILlm` (OpenAI-compatible APIs).
 * The factory `createLlm(settings)` picks by Settings.llm.provider.
 */

import type { Readable } from 'node:stream';

export interface ChatMessage {
  readonly role: 'system' | 'user' | 'assistant' | 'tool';
  readonly content: string;
  readonly name?: string;
  readonly toolCallId?: string;
}

export interface ToolSpec {
  readonly name: string;
  readonly description: string;
  readonly jsonSchema: Readonly<Record<string, unknown>>;
}

export interface GenerateOptions {
  readonly model: string;
  readonly messages: readonly ChatMessage[];
  readonly temperature?: number;
  readonly maxTokens?: number;
  readonly tools?: readonly ToolSpec[];
  readonly responseFormat?: { readonly type: 'json_object' | 'text' } | undefined;
  readonly signal?: AbortSignal;
}

export interface ToolCall {
  readonly name: string;
  readonly args: Readonly<Record<string, unknown>>;
}

export interface GenerateResult {
  readonly content: string;
  readonly toolCalls: readonly ToolCall[];
  readonly usage: { readonly promptTokens: number; readonly completionTokens: number; readonly totalTokens: number };
  readonly finishReason: string;
}

export interface StreamChunk {
  readonly delta: string;
  readonly toolCalls: readonly ToolCall[];
  readonly finishReason: string | null;
}

export interface Llm {
  readonly provider: string;
  readonly model: string;
  generate(opts: GenerateOptions): Promise<GenerateResult>;
  stream(opts: GenerateOptions): AsyncIterable<StreamChunk>;
  /** For clients that need raw stream access (SSE proxies). */
  rawStream(opts: GenerateOptions): Promise<Readable>;
}