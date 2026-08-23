/**
 * Embedder factory + barrel.
 *
 * Picks an embedder based on `Settings.embedder.provider`. Throws
 * `ConfigurationError` for unknown providers or missing api keys.
 */

import { ConfigurationError } from '../errors/index.js';
import type { Settings } from '../settings/index.js';
import { FeatureHashingEmbedder } from './feature-hashing.js';
import { OpenAIEmbedder } from './openai.js';
import type { Embedder } from './types.js';

export type { Embedder } from './types.js';
export { FeatureHashingEmbedder } from './feature-hashing.js';
export { OpenAIEmbedder } from './openai.js';

export const createEmbedder = (settings: Settings): Embedder => {
  switch (settings.embedder.provider) {
    case 'feature_hashing':
      return new FeatureHashingEmbedder(settings.embedder.model, settings.vectorStore.embeddingDim);
    case 'openai': {
      if (!settings.embedder.apiKey) {
        return new FeatureHashingEmbedder(settings.embedder.model, settings.vectorStore.embeddingDim);
      }
      return new OpenAIEmbedder(
        {
          model: settings.embedder.model,
          apiKey: settings.embedder.apiKey,
          batchSize: settings.embedder.batchSize,
        },
        settings.vectorStore.embeddingDim,
      );
    }
    case 'litellm':
    case 'cohere':
      throw new ConfigurationError(
        `embedder provider ${settings.embedder.provider} not yet implemented`,
        { details: { provider: settings.embedder.provider } },
      );
    default: {
      const _exhaustive: never = settings.embedder.provider;
      throw new ConfigurationError(`unknown embedder provider: ${String(_exhaustive)}`);
    }
  }
};