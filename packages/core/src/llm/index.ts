/**
 * LLM factory + barrel.
 *
 * Picks an LLM based on `Settings.llm.provider`. The fallback to the
 * FeatureHashingLlm fires when no API key is present so the rest of
 * the stack keeps running.
 */

import { ConfigurationError } from '../errors/index.js';
import type { Settings } from '../settings/index.js';
import { FeatureHashingLlm } from './feature-hashing.js';
import { OpenAILlm } from './openai.js';
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

export const createLlm = (settings: Settings): Llm => {
  switch (settings.llm.provider) {
    case 'openai': {
      if (!settings.llm.apiKey) return new FeatureHashingLlm(settings.llm.model);
      return new OpenAILlm({
        apiKey: settings.llm.apiKey,
        model: settings.llm.model,
        temperature: settings.llm.temperature,
      });
    }
    case 'litellm': {
      if (!settings.llm.apiKey) return new FeatureHashingLlm(settings.llm.model);
      return new OpenAILlm({
        apiKey: settings.llm.apiKey,
        model: settings.llm.model,
        temperature: settings.llm.temperature,
        baseUrl: 'http://localhost:4000/v1',
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