# Embedders & LLM

`@revex/core` provides embedders and LLM providers with deterministic,
no-network fallbacks so the stack keeps running without API keys.

## Embedders

Interface methods: `model`, `dimension`, `embedQuery(text)`,
`embedDocuments(texts)`.

| Provider | Class | Notes |
|---|---|---|
| OpenAI | `OpenAIEmbedder` | default `text-embedding-3-large`, 3072 dims, lazy-loaded SDK. |
| Feature hashing | `FeatureHashingEmbedder` | deterministic FNV-1a, default 3072 dims, range 64–4096. No network. |
| LiteLLM / Cohere | — | declared providers; `ConfigurationError` (not yet implemented). |

`createEmbedder(settings)` returns `FeatureHashingEmbedder` when no API key is
available, otherwise `OpenAIEmbedder`.

## LLM interface

```ts
interface Llm {
  readonly provider: string;
  readonly model: string;
  generate(opts: GenerateOptions): Promise<GenerateResult>;
  stream(opts: GenerateOptions): AsyncGenerator<StreamChunk>;
  rawStream?(opts): AsyncGenerator<StreamChunk>;
}
```

`GenerateOptions` carries `messages`, `temperature`, `model`, optional
`responseFormat` / tools. `GenerateResult` has `content`, `toolCalls`,
`usage`, `finishReason`.

| Provider | Class | Notes |
|---|---|---|
| OpenAI | `OpenAILlm` | Bearer or Raw `Authorization` header (MiniMax-style). |
| Feature hashing | `FeatureHashingLlm` | no-network fallback. |
| Stub | `StubLlm` | returned when `REVEX_LLM_STUB=1`; deterministic, delay-streaming. |

`createLlm(settings)`:

- Returns `StubLlm` when `REVEX_LLM_STUB=1`.
- `openai`, `minimax`, `litellm` → `OpenAILlm` (falls back to
  `FeatureHashingLlm` without an API key).
  - `minimax` base URL defaults to `https://api.minimax.chat/v1`.
  - `litellm` base URL defaults to `http://localhost:4000/v1`.
- `anthropic` / `bedrock` → `ConfigurationError` (not yet implemented).

## LlmManager

```ts
import { createManager } from '@revex/core';

const manager = createManager({ providers, model });
const result = await manager.generate(messages);
```

`LlmManager` adds retry, a fallback chain across providers, usage aggregation,
and an error taxonomy. It records every provider attempt (`LlmAttemptInfo`) and
only falls back on retryable errors. `LlmUsageTotals` aggregates token usage.

## Errors & token estimates

- `LlmError` + `classifyError(err)` → `LlmErrorKind`.
- `estimateTokens(text)` — `len / 4` heuristic.
- `estimateMessagesTokens(messages)` — per-message estimate.
- `messagesFromConversation(conversation)` — shape turns into chat messages.

## Settings

LLM + embedder config lives in the Zod `Settings` tree:

- `llm.provider`: `openai | minimax | litellm | anthropic | bedrock`.
- `llm.model` (default `gpt-4.1`), `llm.apiKey`, `llm.baseUrl`,
  `llm.temperature` (default `0`).
- `embedder.provider`: `openai | feature_hashing | litellm | cohere`.
- `embedder.model`, `embedder.apiKey`, `embedder.batchSize`.

Loaded via `loadSettings(env)` from `REVEX_LLM_*` / `REVEX_EMBEDDER_*` env
vars (see [API reference env table](../reference/api.md)).