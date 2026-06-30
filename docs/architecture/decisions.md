# Architecture Decision Records

| Number | Date | Title | Status |
|---|---|---|---|
| ADR-0001 | 2024-Q4 | LLM and embedding provider is LiteLLM | Accepted |
| ADR-0002 | 2024-Q4 | Vector store package is `raghub.vectorstore` (singular) | Accepted |
| ADR-0003 | 2024-Q4 | OKF is the canonical persisted knowledge representation | Accepted |
| ADR-0004 | 2024-Q4 | Public API is `raghub.RAG`; legacy services preserved | Accepted |
| ADR-0005 | 2024-Q4 | Telemetry scrubbing is the default | Accepted |
| ADR-0006 | 2024-Q4 | No raw dicts cross the public boundary | Accepted |
| ADR-0007 | 2024-Q4 | Langfuse v3+ uses `get_client()` and spans | Accepted |
| ADR-0008 | 2024-Q4 | Instructor v1+ uses `from_provider("litellm/<model>")` | Accepted |
| ADR-0009 | 2024-Q4 | Configuration precedence: env > TOML > YAML > defaults | Accepted |
| ADR-0010 | 2024-Q4 | Incremental indexing by SHA-256 content hash | Accepted |
| ADR-0011 | 2024-Q4 | Multi-user RBAC enforced at the retrieval layer | Accepted |
| ADR-0012 | 2024-Q4 | Conversation history is namespaced by `user + session_id` | Accepted |
| ADR-0013 | 2024-Q4 | Streaming goes end-to-end through the LLM's `astream` | Accepted |
| ADR-0014 | 2024-Q4 | Plugin registry is type-keyed and entry-point discoverable | Accepted |
| ADR-0015 | 2024-Q4 | Structured output is delivered via Instructor when available | Accepted |
| ADR-0016 | 2024-Q4 | Background ingestion is resumable through a persistent ledger | Accepted |
| ADR-0017 | 2024-Q4 | Production forbids passwordless login and short JWT secrets | Accepted |

---

## ADR-0001: LLM and embedding provider is LiteLLM

**Context:** the previous default was `langchain-nvidia-ai-endpoints`,
which is a thin wrapper around NVIDIA's hosted models and locks the
project into a single provider.

**Decision:** the canonical LLM and embedding providers are
`raghub.llm.litellm.LiteLLMProvider` and
`raghub.embeddings.litellm.LiteLLMEmbeddingProvider`. Both work with
any LiteLLM-supported model (OpenAI, Anthropic, NVIDIA, Bedrock,
Cohere, Voyage, Groq).

**Consequences:** the framework supports multiple providers out of
the box; the dependency on `langchain-nvidia-ai-endpoints` was
removed. When no API key is configured, `default_llm` /
`default_embedder` fall back to the offline
`HeuristicLLMProvider` / `HashingEmbeddingProvider` so the facade
runs anywhere.

## ADR-0002: Vector store package is `raghub.vectorstore` (singular)

**Context:** two parallel packages (`raghub.vectorstore` singular
and `raghub.vectorstores` plural) confused importers.

**Decision:** one package: `raghub.vectorstore`. Qdrant lives at
`raghub.vectorstore.qdrant`; ZVec at `raghub.vectorstore.zvec`;
InMemory at `raghub.vectorstore.memory`.

**Consequences:** the duplicate plural package was merged; legacy
imports of `raghub.vectorstores.*` were redirected.

## ADR-0003: OKF is the canonical persisted knowledge representation

**Context:** the previous pipeline produced `KnowledgeBundle` only
in memory; there was no canonical on-disk schema for hand-off
between pipelines.

**Decision:** the canonical persisted representation is **OKF
(Open Knowledge Format)**: a `KnowledgeBundle` of
`DocumentSection` × `DocumentBlock`, round-tripped through
`to_okf` / `from_okf`.

**Consequences:** bundles can be cached, replayed, and inspected by
third-party tools. The `InMemoryKnowledgeRepository` lives at
`raghub.knowledge.repository`; a disk-backed variant is an obvious
extension point.

## ADR-0004: Public API is `raghub.RAG`; legacy services preserved

**Context:** the legacy `DynamicRagApplication`/`build_application`
surface tied auth, storage, and pipelines to a single use-case
facade — too many concerns in one import.

**Decision:** the recommended entry point is `raghub.RAG`, a small
DI container that wires pipelines against replaceable adapters.
The legacy `DynamicRagApplication` and `build_application` remain
reachable from `raghub.__init__` for backward compatibility and
power the FastAPI app at `raghub.api.app:app`.

**Consequences:** new code uses `RAG(...)`. Legacy endpoints, CLI
shims, and DI containers remain available so existing consumers
do not break.

## ADR-0005: Telemetry scrubbing is the default

**Context:** every `TelemetryProvider` typically sees kwargs that
contain credentials, tokens, and PII; without scrubbing the
upstream sink (e.g. Langfuse) sees raw passwords in event
attributes.

**Decision:** the facade wraps its default telemetry in
`RedactingTelemetry`, which scrubs any kwarg whose key matches
`password|secret|api_key|token|jwt|authorization` (case-insensitive,
recursive into nested dicts) before forwarding.

**Consequences:** secret leakage to telemetry sinks is blocked at
the facade boundary. The scrubbing is opt-out (pass your own
`telemetry=` to disable).

## ADR-0006: No raw dicts cross the public boundary

**Context:** unrestrained `dict[str, Any]` return types break
typed callers, encourage `Any` annotations, and complicate docs.

**Decision:** every public API in the facade (`RAG`, the
`IngestPipeline`, the `QueryPipeline`, the CLI, the FastAPI
schemas) exchanges typed Pydantic models declared in
`raghub.models`.

**Consequences:** refactors stay non-breaking; static type
checking is meaningful; OpenAPI schemas are generated from real
Pydantic models.

## ADR-0007: Langfuse v3+ uses `get_client()` and spans

**Context:** the previous `LangfuseTelemetryProvider` used the
v2 SDK (`Langfuse(...)` + `langfuse.score(...)` for every event)
and produced no proper spans.

**Decision:** the provider uses langfuse v3+ (`get_client()` +
`start_as_current_observation(as_type="span"|"generation", ...)`).
The provider degrades to `NoOpTelemetry` when langfuse is missing
or unconfigured.

**Consequences:** traces are first-class observations; the
`RAG` facade emits per-stage spans, latency, and token usage.
The dependency on langfuse v2 API patterns was removed.

## ADR-0008: Instructor v1+ uses `from_provider("litellm/<model>")`

**Context:** Instructor had two styles of API depending on
provider; the older `instructor.from_litellm(...)` wrapper was
deprecated.

**Decision:** the structured-output provider initialises the
Instructor client via `instructor.from_provider("litellm/<model>")`.

**Consequences:** the call shape is uniform across LiteLLM
providers; Instructor upgrades are absorbed behind the
`StructuredOutputProvider` interface.

## ADR-0009: Configuration precedence: env > TOML > YAML > defaults

**Context:** configuration precedence was implicit (YAML ↦ env
override), confusing contributors reading `load_settings`.

**Decision:** the explicit precedence is **env > TOML > YAML >
defaults**. The facade's `RAG.from_config` accepts both `.yaml`
and `.toml`; TOML is merged on top of YAML when both exist.

**Consequences:** operators have a deterministic rule for "what
wins?". Free-form keys are preserved on `settings.extra` for
forward compatibility.

## ADR-0010: Incremental indexing by SHA-256 content hash

**Context:** re-ingesting an unchanged document wastes LLM,
embedder, and vector-store I/O; downstream indexes are not
idempotent.

**Decision:** `IngestPipeline` computes `sha256(file_bytes)` and
asks `KnowledgeRepository.get(bundle_id)` for a prior checksum.
If the prior checksum matches, no re-embedding happens; the
prior chunks are returned with `outputs["incremental"] = True`.

**Consequences:** `rag.ingest` is idempotent on the source.
`rag.sync_index(directory)` reconciles an entire directory in one
call and emits `added`/`modified`/`unchanged`/`removed` lists.

## ADR-0011: Multi-user RBAC enforced at the retrieval layer

**Context:** previous code performed retrieval with *no* filter
and attempted to scrub results in post — a leaky abstraction
that allowed unauthorised context to reach the prompt.

**Decision:** `QueryPipeline.metadata_filter_for_user(user)`
derives a metadata filter from `user.allowed_companies`
(direct; admins return `""`; users with empty allow-lists return
`{"company": []}` which matches nothing). The filter is forwarded
to `vector_store.search`. The LLM only ever sees what the
filter returned.

**Consequences:** there is no longer a code path by which
unauthorised content can reach the prompt. Users with empty
allow-lists receive empty result sets, not 403 errors.

## ADR-0012: Conversation history is namespaced by `user + session_id`

**Context:** `session_id` alone could be guessed; two users
sharing an id would otherwise read each other's history.

**Decision:** `RAG._scoped_session_id(user, session_id)` returns
`f"{uid}::{session_id}"` (or the raw id when no user is set, for
the in-process anonymous case). `conversation_history` and
`clear_conversation` likewise scope by the combined key.

**Consequences:** conversation history cannot leak across users
who happen to share a `session_id`.

## ADR-0013: Streaming goes end-to-end through the LLM's `astream`

**Context:** earlier versions of `RAG.astream` materialised the
full answer before yielding chunks.

**Decision:** `QueryPipeline.stream` calls
`Generator.astream` → `DefaultGenerator.astream` →
`LiteLLMProvider.astream`, with
`stream_options={"include_usage": True}` so token usage is
populated as the stream completes.

**Consequences:** time-to-first-byte is dominated by the LLM's
first token, not by assembling the full answer. The
`record_tokens()` hook emits per-stream usage to telemetry.

## ADR-0014: Plugin registry is type-keyed and entry-point discoverable

**Context:** plugins with a global namespace were ambiguous and
hard to reason about.

**Decision:** `PluginRegistry` keeps separate mappings for
converters, chunkers, embedders, vector stores, knowledge
repos, generators, structured providers, telemetry, evaluators,
and generic factories. Plugins may register through
`register_*` helpers or via the `raghub.plugins` entry-point
group (`PluginRegistry.discover_entrypoints()` loads them).

**Consequences:** every collaborator behind the facade is
swappable through the same API. Plugins may add components
without modifying the framework source.

## ADR-0015: Structured output is delivered via Instructor when available

**Context:** call sites that wanted a typed answer were rolling
their own regex extraction.

**Decision:** `RAG.query(..., response_model=MyModel)` returns
the typed `BaseModel` in `Response.structured`. The structured
path requires Instructor + an LLM API key; otherwise the field
is `None` and the call still returns the free-form answer.

**Consequences:** typed answers are a first-class capability; the
graceful degradation keeps `RAG` usable offline.

## ADR-0016: Background ingestion is resumable through a persistent ledger

**Context:** in-process background ingestion lost state on
crash; pending jobs were silently dropped.

**Decision:** `ResumableBackgroundIngestionService` writes every
status transition to a SQLite ledger via `PersistentJobStore`.
On startup, `restore_from_store()` rehydrates the in-memory map.

**Consequences:** a crash mid-ingestion does not lose queued
work. `rag.ingest_async(...)` and `rag.job_status(job_id)` round-
trip through the same ledger.

## ADR-0017: Production forbids passwordless login and short JWT secrets

**Context:** the legacy `allow_passwordless_login: true` default
and PyJWT's `InsecureKeyLengthWarning` for sub-32-byte secrets
were footguns.

**Decision:** `load_settings` enforces, when
`environment == "production"`, that `JWT_SECRET` is set, is at
least 32 UTF-8 bytes long, and that `allow_passwordless_login`
is `false`. Any violation raises `RuntimeError` at startup.

**Consequences:** the production profile cannot silently accept
forged credentials or permit passwordless sessions. CI also
fails the build when the `InsecureKeyLengthWarning` fires.
