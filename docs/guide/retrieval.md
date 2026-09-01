# Retrieval & reranking

Revex retrieves with a hybrid pipeline: dense vector search + BM25 keyword
search, fused by **Reciprocal Rank Fusion** (RRF), scoped by RBAC.

## `Retrieval` pipeline

```ts
import { Retrieval } from '@revex/core';

const retrieval = new Retrieval({ embedder, vectorStore, opts });
const hits = await retrieval.retrieve(user, question);
```

`RetrievalOptions` configures the fusion weights (`denseWeight`,
`sparseWeight`, `rrfK`), top-K, the reranker, and whether ColBERT
late-interaction scoring is applied.

## Fusion

- `reciprocalRankFusion(items, k)` — RRF: `score = Σ 1/(k + rank)`. Default
  `k = 60`.
- `linearFusion(items, { denseWeight, sparseWeight })` — a weighted linear
  blend. `DEFAULT_LINEAR` is `0.6 / 0.4`.
- `lateInteractionScore` / `lateInteractionRerank` — ColBERT-style MaxSim
  late interaction.

## Rerankers

Rerankers are registered polymorphically via `RerankerFactory`:

| Reranker | Key | Description |
|---|---|---|
| `IdentityReranker` | `identity` | Pass-through (default). |
| `CohereReranker` | `cohere` | Delegates to the Cohere rerank API. |
| `LlmReranker` | `llm_judge` | Uses an LLM to score relevance. |

`registerBuiltInRerankers()` registers the defaults.

## Transformers

`@revex/core` ships retrieval transformers:

| Transformer | Purpose |
|---|---|
| `createHydeTransformer` | Hypothetical Document Embeddings (HyDE). |
| `createMultiQueryTransformer` | Expand one query into several. |
| `createStepBackTransformer` | Ask a broader "step-back" question. |
| `createDecomposeTransformer` | Break a query into sub-questions. |
| `createComposeTransformer` | Merge sub-answers. |
| `createCascadeRouter` + `CascadeStages` | Route between cascaded retrieval stages. |

## Feedback scorers

Feedback influences future retrieval:

- `Bm25BoostScorer` — boosts chunks with positive ratings, down-weights
  negatively-rated terms.
- `VectorDownWeightScorer` — down-weights similar future queries.
- `NoOpFeedbackScorer` — no-op.

`registerBuiltInScorers()` registers them under the
`revex.feedback_scorers` plugin group.

## Context building

`buildContext(input)` shapes retrieved hits + conversation turns into an LLM
context block under a token budget. `defaultBudget(model)` returns the budget;
`ContextBuildStats` summarizes truncation and source citations. `summariseContext`
renders a short stats line.

## Embedders

- `OpenAIEmbedder` — defaults to `text-embedding-3-large`, 3072 dims,
  batch size 32.
- `FeatureHashingEmbedder` — deterministic FNV-1a fallback (no network).
- `createEmbedder(settings)` picks the provider (falls back to feature-hashing
  when no API key).

## RBAC filter

`StoreFilter` and `allowedCompanyFilter(user)` scope every retrieval by
workspace, user, collection, principals, and allowed companies. The vector
store enforces ACLs in SQL.