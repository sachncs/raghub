/**
 * LLM factory + barrel.
 *
 * Picks an LLM based on `Settings.llm.provider`. The fallback to the
 * FeatureHashingLlm fires when no API key is present so the rest of
 * the stack keeps running.
 *
 * `minimax` (https://platform.minimax.io/docs/guides/models-intro)
 * is OpenAI-compatible and routed via `OpenAILlm` with a configurable
 * `baseUrl` (defaults to `https://api.minimax.chat/v1`).
 *
 * When `REVEX_LLM_STUB=1`, `createLlm` returns a `StubLlm`
 * regardless of provider — used by the web smoke suite and by
 * local runs that don't have an API key handy.
 */

import { ConfigurationError } from '../errors/index.js';
import type { Settings } from '../settings/index.js';
import { FeatureHashingLlm } from './feature-hashing.js';
import { OpenAILlm } from './openai.js';
import { StubLlm } from './stub.js';
import type { Llm } from './types.js';

export type {
  ChatMessage,
  GenerateOptions,
  GenerateResult,
  Llm,
  StreamChunk,
  ToolCall,
  ToolSpec,
} from './types.js';
export { OpenAILlm } from './openai.js';
export { FeatureHashingLlm } from './feature-hashing.js';
export { StubLlm } from './stub.js';
export type { StubLlmOptions } from './stub.js';

const MINIMAX_BASE_URL = 'https://api.minimax.chat/v1';

export const createLlm = (settings: Settings): Llm => {
  if (process.env['REVEX_LLM_STUB'] === '1') {
    return new StubLlm({ model: settings.llm.model });
  }
  switch (settings.llm.provider) {
    case 'openai': {
      if (!settings.llm.apiKey) return new FeatureHashingLlm(settings.llm.model);
      return new OpenAILlm({
        apiKey: settings.llm.apiKey,
        model: settings.llm.model,
        temperature: settings.llm.temperature,
        ...(settings.llm.baseUrl !== undefined ? { baseUrl: settings.llm.baseUrl } : {}),
      });
    }
    case 'minimax': {
      if (!settings.llm.apiKey) return new FeatureHashingLlm(settings.llm.model);
      return new OpenAILlm({
        apiKey: settings.llm.apiKey,
        model: settings.llm.model,
        temperature: settings.llm.temperature,
        baseUrl: settings.llm.baseUrl ?? MINIMAX_BASE_URL,
      });
    }
    case 'litellm': {
      if (!settings.llm.apiKey) return new FeatureHashingLlm(settings.llm.model);
      return new OpenAILlm({
        apiKey: settings.llm.apiKey,
        model: settings.llm.model,
        temperature: settings.llm.temperature,
        baseUrl: settings.llm.baseUrl ?? 'http://localhost:4000/v1',
      });
    }
    case 'anthropic':
    case 'bedrock':
      throw new ConfigurationError(
        `llm provider ${settings.llm.provider} not yet implemented`,
        { details: { provider: settings.llm.provider } },
      );
    default: {
      const _exhaustive: never = settings.llm.provider;
      throw new ConfigurationError(`unknown llm provider: ${String(_exhaustive)}`);
    }
  }
};