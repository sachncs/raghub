# Migration Guide

This guide covers the changes between the legacy single-service
RAG application and the new `raghub.RAG` facade.

## Public API

### Before

```python
from raghub.core.container import build_application
from raghub.services.application import DynamicRagApplication

app = asyncio.run(build_application())
app.upload_document(token=..., filename=..., content=...)
result = app.query(token=..., question=...)
```

### After

```python
from raghub import RAG

rag = RAG()
rag.ingest("path/to/doc.pdf")
result = rag.query("What is the revenue?")
print(result.answer)
```

## Configuration

### Before

```python
from raghub.config.settings import load_settings
settings = load_settings(profile="dev")
```

### After

```python
from raghub import RAG
rag = RAG.from_config("raghub.yaml")
# or with TOML:
rag = RAG.from_config("raghub.toml")
# or with runtime overrides:
settings = load_settings(profile="dev")
settings = settings.override(chunk_size_words=400)
rag = RAG(settings=settings)
```

## Vector Store Packages

### Before

Two parallel packages: `raghub.vectorstore` (singular, legacy) and
`raghub.vectorstores` (plural, new).

### After

A single package: `raghub.vectorstore` (singular). The Qdrant
adapter moved from `raghub.vectorstores.qdrant` to
`raghub.vectorstore.qdrant`.

## Prompt Builders

### Before

Two parallel classes: `raghub.prompts.builder.PromptBuilder` (used
in production) and `raghub.prompts.builder.TemplatePromptBuilder`
(legacy stub).

### After

A single class: `raghub.prompts.builder.PromptBuilder`.
`TemplatePromptBuilder` is removed. New code should rely on
`DefaultGenerator` (in `raghub.generation.generator`).

## LLM Providers

### Before

`raghub.llm.nvidia.NvidiaLLMProvider` (deleted) was a thin wrapper
around the langchain NVIDIA SDK.

### After

`raghub.llm.litellm.LiteLLMProvider` is the canonical LLM provider.
It works with any LiteLLM-supported model (OpenAI, NVIDIA,
Anthropic, Bedrock, …). Update any direct imports of
`NvidiaLLMProvider` to `LiteLLMProvider(model="nvidia/<model>")`.

## Embedding Providers

### Before

`raghub.embeddings.nvidia.NvidiaEmbeddingProvider` (deleted).

### After

`raghub.embeddings.litellm.LiteLLMEmbeddingProvider` is the canonical
embedding provider. Update any direct imports of
`NvidiaEmbeddingProvider` to
`LiteLLMEmbeddingProvider(model="nvidia/<model>")`.

## Telemetry

### Before

`LangfuseTelemetryProvider` used the v2 SDK and called
`langfuse.score()` for every log event.

### After

The provider uses the documented Langfuse v3+ API
(`get_client()` and `start_as_current_observation`) and emits proper
spans. The default `RAG(...)` constructor automatically wires
Langfuse as the default telemetry provider (when credentials are
present).

## CLI

### Before

```bash
python -m raghub.cli login EMAIL PASSWORD
python -m raghub.cli health
```

### After

```bash
python -m raghub.cli health
python -m raghub.cli ingest ./documents
python -m raghub.cli query "What is the revenue?"
python -m raghub.cli eval financebench --examples 25
```

The legacy `login` / `health` commands remain available via the
`raghub.cli_legacy` shim, but new code should call
`DynamicRagApplication` directly when auth is required.

## Console Scripts

### Before

`raghub-financebench` pointed at `evaluate_financebench:main`,
which is `async def main` and would fail at install time.

### After

`raghub-financebench` points at `raghub.cli.eval_cmd:main` (a sync
shim). It works out of the box.

## Exceptions

The exception hierarchy was extended. The legacy aliases
(`DynamicRagError`, `DocumentError`, `IndexingError`, `PromptError`,
`LLMError`, `StorageError`, `AuthenticationError`,
`AuthorizationError`) are preserved and subclass the new
`RagHubError` base. New code should use the new names
(`ConfigurationError`, `ConversionError`, `KnowledgeError`,
`IngestionError`, `EmbeddingError`, `VectorStoreError`,
`RetrievalError`, `GenerationError`, `PipelineError`,
`EvaluationError`).
