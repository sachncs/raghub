# Architecture Decision Records

| Number | Date       | Title                                                    | Status     |
|--------|------------|----------------------------------------------------------|------------|
| ADR-0001 | 2024-Q4    | LLM and embedding provider is LiteLLM                    | Accepted   |
| ADR-0002 | 2024-Q4    | Vector store package is `raghub.vectorstore` (singular)  | Accepted   |
| ADR-0003 | 2024-Q4    | OKF is the canonical persisted knowledge representation  | Accepted   |
| ADR-0004 | 2024-Q4    | Public API is `raghub.RAG`; legacy services preserved    | Accepted   |
| ADR-0005 | 2024-Q4    | Telemetry scrubbing is the default                        | Accepted   |
| ADR-0006 | 2024-Q4    | No raw dicts cross the public boundary                    | Accepted   |
| ADR-0007 | 2024-Q4    | Langfuse v3+ uses `get_client()` and spans               | Accepted   |
| ADR-0008 | 2024-Q4    | Instructor v1+ uses `from_provider("litellm/<model>")`    | Accepted   |
| ADR-0009 | 2024-Q4    | Configuration precedence: env > TOML > YAML > defaults   | Accepted   |
| ADR-0010 | 2024-Q4    | Incremental indexing by SHA-256 content hash             | Accepted   |

## ADR-0001: LLM and embedding provider is LiteLLM

**Context:** the previous default was `langchain-nvidia-ai-endpoints`,
which is a thin wrapper around NVIDIA's hosted models and locks the
project into a single provider.

**Decision:** the canonical LLM and embedding providers are
`raghub.llm.litellm.LiteLLMProvider` and
`raghub.embeddings.litellm.LiteLLMEmbeddingProvider`. Both work
with any LiteLLM-supported model (OpenAI, NVIDIA, Anthropic,
Bedrock, Cohere, Voyage, …).

**Consequences:** the framework supports multiple providers
out of the box; the dependency on `langchain-nvidia-ai-endpoints`
was removed.

## ADR-0007: Langfuse v3+ uses `get_client()` and spans

**Context:** the previous `LangfuseTelemetryProvider` used the
langfuse v2 SDK (`Langfuse(...)` and `langfuse.score(...)` for
every event) and produced no proper spans.

**Decision:** the provider uses langfuse v3+ (`get_client()` and
`start_as_current_observation(as_type="span"|"generation", ...)`).
The provider degrades to `NoOpTelemetry` when langfuse is not
installed or no credentials are present in the environment.

**Consequences:** traces are first-class observations; the RAG
facade emits per-stage spans, latency, and token usage. The
dependency on langfuse v2 API patterns was removed.
