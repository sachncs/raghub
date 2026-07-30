# Roadmap

> RAGHub v0.5 has shipped as the "renames + structure" release. The
> core facade is stable; further work focuses on operational
> robustness, deeper offline support, and integrations that don't
> add new first-class adapters.

## In Progress

- **`raghub.RAG` in FastAPI.** The current FastAPI app
  (`raghub.api:get_app`) is bound to the legacy `Facade`. A v2 will
  mount the new facade behind thin route handlers, taking
  advantage of `User` at the request boundary.
- **Disk-backed `KnowledgeRepository`.** The current `MemoryRepo`
  is a starting point; an OKF-on-disk variant would let bundles
  survive restarts and be inspected offline.
- **Group ABAC.** `User.allowed_groups` is already in the model.
  Wiring groups through the retrieval filter is straightforward;
  the contract is in `docs/architecture/decisions.md`.
- **Pluggable rerankers.** `Identity` ships by default; a
  cross-encoder reranker behind the same `Rerank` protocol would
  be a single plugin.
- **Async batching in `Ingestor`.** The current ingest path is
  fully sequential per file; a small concurrent pool would cut
  end-to-end latency for directory ingests.

## Considered — Open Questions

- **Persistent conversation store.** The default
  `MemoryConversations` is fine for a single process. For
  multi-worker deployments behind a load balancer, a SQLite-backed
  store that respects the `ConversationStore` protocol is needed
  (it would be a plugin, no facade change).
- **Annotation loops.** Persisting user feedback on chunks
  (thumbs up/down on citations) for offline re-ranking.
- **Multi-tenant database isolation.** Multi-database-per-tenant
  is more invasive than the current `company` filter; deferred
  until request volume justifies the operational complexity.

## Probably Not (Without a Strong Reason)

- New embedding providers. LiteLLM already routes to OpenAI,
  Cohere, Voyage, NVIDIA, HuggingFace, etc. — write a plugin for
  anything else.
- New LLM providers for the same reason; LiteLLM covers OpenAI,
  Anthropic, NVIDIA, Bedrock, Cohere, Voyage, Groq, and Ollama.
- A remote vector store. `SqliteStore` is the spec default; the
  `Store` interface already lets you wire FAISS, Chroma, pgvector,
  or any backend behind a plugin.

## Performance Follow-Ups

- The CLI benchmark (`python -m devtools.benchmark`) measures
  startup time, ingestion throughput, query latency (p50/p95),
  queries per second, and peak RSS. Use it to baseline before
  adding caching, batching, or streaming changes.
- A Redis-backed caching layer for embeddings (and LLM responses,
  where the prompt+context is small) is an obvious next step —
  implement it behind a tiny protocol and inject into the `RAG`
  facade.

## Observability Follow-Ups

- Token-level cost attribution per `session_id` is already feasible
  from `LiteLLM.last_usage`. Persisting those counters in a
  structured log sink and surfacing in Langfuse is the next step.
- Trace sampling. The current `RedactingTelemetry` wraps every
  call; a sampler that drops low-value spans would reduce cost on
  the Langfuse side.