# Roadmap

> The framework already wires the spec libraries as defaults
> (Marker, Chonkie, LiteLLM, Instructor, Qdrant, Langfuse). Future
> work focuses on plugins, performance, and operational robustness
> rather than adding new first-class adapters.

## In Progress

- **`raghub.RAG` in FastAPI.** The current FastAPI app
  (`raghub.api.app:app`) is bound to the legacy
  `DynamicRagApplication`. A v2 will mount the new facade behind
  thin route handlers, taking advantage of `UserPrincipal` at
  the request boundary.
- **Disk-backed `KnowledgeRepository`.** The current
  `InMemoryKnowledgeRepository` is a starting point; an OKF-on-
  disk variant would let bundles survive restarts and be
  inspected offline.
- **`SecureReadUser` / group ABAC.** `UserPrincipal.allowed_groups`
  is already in the model. Wiring groups through the retrieval
  filter is straightforward; the contract is in
  `docs/architecture/decisions.md`.
- **Typed rerankers.** `IdentityReranker` ships by default; a
  cross-encoder reranker behind the same `Reranker` interface
  would be a single plugin.

## Considered — Open Questions

- **SQL-backed conversation store.** The default
  `InMemoryConversationStore` is fine for a single process. For
  multi-worker deployments behind a load balancer, a Redis- or
  SQLite-backed store that respects the `ConversationStore`
  protocol is needed (it would be a plugin, no facade change).
- **Annotation loops.** Persisting user feedback on chunks (thumbs
  up/down on citations) for offline re-ranking.
- **Multi-tenant database isolation.** Multi-database-per-tenant
  is more invasive than the current `company` filter; deferred
  until request volume justifies the operational complexity.

## Probably Not (Without a Strong Reason)

- New embedding providers. LiteLLM already routes to OpenAI,
  Cohere, Voyage, NVIDIA, HuggingFace, etc. — write a plugin
  for anything else.
- New LLM providers for the same reason; LiteLLM covers OpenAI,
  Anthropic, NVIDIA, Bedrock, Cohere, Voyage, Groq, and Ollama.
- New vector stores for the same reason; Qdrant is the spec
  default, and the `VectorStore` interface already lets you wire
  Milvus, FAISS, Chroma, or pgvector behind a plugin.

## Performance Follow-Ups

- The CLI benchmark (`python -m bench.benchmark`) measures startup
  time, ingestion throughput, query latency (p50/p95), queries
  per second, and peak RSS. Use it to baseline before adding
  caching, batching, or streaming changes.
- A Redis-backed caching layer for embeddings (and LLM
  responses, where the prompt+context is small) is an obvious
  next step — implement it behind a tiny protocol and inject
  into the `RAG` facade.

## Observability Follow-Ups

- Token-level cost attribution per `session_id` is already
  feasible from `LiteLLMProvider.last_usage`. Persisting those
  counters in a structured log sink and surfacing in Langfuse is
  the next step.
- Trace sampling. The current `RedactingTelemetry` wraps every
  call; a sampler that drops low-value spans would reduce
  cost on the Langfuse side.
