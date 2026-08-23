/**
 * OpenAI-compatible LLM.
 *
 * Works against the OpenAI API directly or any compatible endpoint
 * (Together, OpenRouter, vLLM, Ollama's /v1, etc.) — pass a custom
 * `baseUrl` and it works the same way.
 *
 * The SDK is loaded lazily so the package builds without it; the
 * first call throws MissingDepError if the import fails.
 */

import { Readable } from 'node:stream';

import { ConfigurationError, MissingDepError } from '../errors/index.js';
import type {
  ChatMessage,
  GenerateOptions,
  GenerateResult,
  Llm,
  StreamChunk,
  ToolCall,
  ToolSpec,
} from './types.js';

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;
  tool_call_id?: string;
  tool_calls?: { id: string; type: 'function'; function: { name: string; arguments: string } }[];
}

interface OpenAITool {
  type: 'function';
  function: { name: string; description: string; parameters: Record<string, unknown> };
}

interface OpenAIChatRequest {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  tools?: OpenAITool[];
  response_format?: { type: 'json_object' | 'text' };
  stream?: boolean;
}

interface OpenAIChatResponse {
  choices: { message: OpenAIMessage & { tool_calls?: OpenAIMessage['tool_calls'] }; finish_reason: string; index: number }[];
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

interface OpenAIChatChunk {
  choices: {
    delta: { content?: string; tool_calls?: { index: number; id?: string; type?: 'function'; function?: { name?: string; arguments?: string } }[] };
    finish_reason: string | null;
  }[];
}

interface OpenAIClient {
  chat: {
    completions: {
      create: (req: OpenAIChatRequest) => Promise<OpenAIChatResponse>;
    };
  };
}

interface OpenAIModule {
  default: new (opts: { apiKey: string; baseURL?: string }) => OpenAIClient;
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadOpenAI = async (): Promise<OpenAIModule> => {
  try {
    const mod = (await dynamicImport('openai')) as { default?: OpenAIModule } & OpenAIModule;
    return mod.default ?? mod;
  } catch (cause) {
    throw new MissingDepError('openai package is not installed', {
      cause,
      details: { hint: 'pnpm add openai' },
    });
  }
};

const mapMessages = (msgs: readonly ChatMessage[]): OpenAIMessage[] =>
  msgs.map((m): OpenAIMessage => {
    const out: OpenAIMessage = { role: m.role, content: m.content };
    if (m.name) out.name = m.name;
    if (m.toolCallId) out.tool_call_id = m.toolCallId;
    return out;
  });

const mapTools = (tools: readonly ToolSpec[]): OpenAITool[] =>
  tools.map((t) => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: { ...t.jsonSchema },
    },
  }));

const parseToolCalls = (calls: OpenAIChatResponse['choices'][number]['message']['tool_calls']): ToolCall[] => {
  if (!calls) return [];
  const out: ToolCall[] = [];
  for (const c of calls) {
    try {
      out.push({ name: c.function.name, args: JSON.parse(c.function.arguments) });
    } catch {
      out.push({ name: c.function.name, args: {} });
    }
  }
  return out;
};

export interface OpenAILlmOptions {
  readonly apiKey: string;
  readonly baseUrl?: string;
  readonly model: string;
  readonly temperature?: number;
  /**
   * Some providers (notably MiniMax) require a custom
   * `Authorization` header prefix (e.g. `Bearer` vs raw API key).
   * Default is `Bearer`.
   */
  readonly authorizationPrefix?: 'Bearer' | 'Raw';
}

export class OpenAILlm implements Llm {
  public readonly provider = 'openai';
  public readonly model: string;
  private readonly apiKey: string;
  private readonly baseUrl: string | undefined;
  private readonly temperature: number;
  private readonly authorizationPrefix: 'Bearer' | 'Raw';
  private clientPromise: Promise<OpenAIClient> | null = null;

  constructor(opts: OpenAILlmOptions) {
    if (!opts.apiKey) throw new ConfigurationError('openai llm requires apiKey');
    this.apiKey = opts.apiKey;
    this.baseUrl = opts.baseUrl;
    this.model = opts.model;
    this.temperature = opts.temperature ?? 0;
    this.authorizationPrefix = opts.authorizationPrefix ?? 'Bearer';
  }

  public async generate(opts: GenerateOptions): Promise<GenerateResult> {
    const client = await this.ensureClient();
    const req = this.buildRequest(opts, false);
    const resp = await client.chat.completions.create(req);
    const choice = resp.choices[0];
    if (!choice) throw new ConfigurationError('openai returned no choices');
    const content = choice.message.content ?? '';
    const toolCalls = parseToolCalls(choice.message.tool_calls);
    const usage = resp.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    return {
      content,
      toolCalls,
      usage: {
        promptTokens: usage.prompt_tokens,
        completionTokens: usage.completion_tokens,
        totalTokens: usage.total_tokens,
      },
      finishReason: choice.finish_reason,
    };
  }

  public async *stream(opts: GenerateOptions): AsyncIterable<StreamChunk> {
    const stream = await this.rawStream(opts);
    let buffer = '';
    for await (const raw of streamToChunks(stream)) {
      const parsed = parseSseLine(raw);
      if (!parsed) continue;
      buffer += parsed;
      const events = splitSseEvents(buffer);
      for (const ev of events) {
        if (!ev || !ev.startsWith('data:')) continue;
        const data = ev.slice(5).trim();
        if (data === '[DONE]') {
          yield { delta: '', toolCalls: [], finishReason: 'stop' };
          return;
        }
        try {
          const obj = JSON.parse(data) as OpenAIChatChunk;
          const c = obj.choices[0];
          if (!c) continue;
          const toolCalls: ToolCall[] = [];
          if (c.delta.tool_calls) {
            for (const tc of c.delta.tool_calls) {
              if (tc.function?.name) {
                toolCalls.push({
                  name: tc.function.name,
                  args: safeJsonParse(tc.function.arguments ?? '{}'),
                });
              }
            }
          }
          yield {
            delta: c.delta.content ?? '',
            toolCalls,
            finishReason: c.finish_reason,
          };
        } catch {
          // ignore malformed chunks
        }
      }
      buffer = '';
    }
  }

  public async rawStream(opts: GenerateOptions): Promise<Readable> {
    const mod = await loadOpenAI();
    if (!this.clientPromise) {
      const client = new mod.default({
        apiKey: this.apiKey,
        ...(this.baseUrl ? { baseURL: this.baseUrl } : {}),
      });
      this.clientPromise = Promise.resolve(client);
    }
    const client = await this.clientPromise;
    const req = this.buildRequest(opts, true);
    const raw = await (client as unknown as {
      chat: {
        completions: {
          create: (
            req: OpenAIChatRequest,
            opts: { responseType?: 'arraybuffer' | 'stream' },
          ) => Promise<unknown>;
        };
      };
    }).chat.completions.create(req, { responseType: 'stream' });
    if (raw instanceof Readable) return raw;
    if (raw && typeof (raw as { getReader?: unknown }).getReader === 'function') {
      return Readable.fromWeb(raw as unknown as import('node:stream/web').ReadableStream);
    }
    throw new ConfigurationError('openai stream response was not a Readable');
  }

  private buildRequest(opts: GenerateOptions, stream: boolean): OpenAIChatRequest {
    const req: OpenAIChatRequest = {
      model: opts.model,
      messages: mapMessages(opts.messages),
      temperature: opts.temperature ?? this.temperature,
      ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
      ...(opts.tools && opts.tools.length > 0 ? { tools: mapTools(opts.tools) } : {}),
      ...(opts.responseFormat !== undefined ? { response_format: opts.responseFormat } : {}),
      stream,
    };
    return req;
  }

  private async ensureClient(): Promise<OpenAIClient> {
    if (this.clientPromise) return this.clientPromise;
    const mod = await loadOpenAI();
    const authHeader =
      this.authorizationPrefix === 'Raw' ? this.apiKey : `Bearer ${this.apiKey}`;
    const ctor = mod.default as unknown as new (opts: {
      apiKey: string;
      baseURL?: string;
      defaultHeaders?: Record<string, string>;
    }) => OpenAIClient;
    const client = new ctor({
      apiKey: this.apiKey,
      defaultHeaders: { Authorization: authHeader },
      ...(this.baseUrl ? { baseURL: this.baseUrl } : {}),
    });
    this.clientPromise = Promise.resolve(client);
    return client;
  }
}

async function* streamToChunks(stream: Readable): AsyncIterable<string> {
  let pending = '';
  for await (const piece of stream) {
    pending += String(piece);
    let idx;
    while ((idx = pending.indexOf('\n\n')) >= 0) {
      yield pending.slice(0, idx + 2);
      pending = pending.slice(idx + 2);
    }
  }
  if (pending) yield pending;
}

function parseSseLine(raw: string): string | null {
  if (!raw) return null;
  const trimmed = raw.endsWith('\n\n') ? raw : raw + '\n\n';
  return trimmed;
}

function splitSseEvents(buffer: string): string[] {
  return buffer.split('\n\n').filter((e) => e.length > 0);
}

function safeJsonParse(s: string): Readonly<Record<string, unknown>> {
  try {
    return JSON.parse(s) as Record<string, unknown>;
  } catch {
    return {};
  }
}