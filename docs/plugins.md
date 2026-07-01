# Plugin Development

A **plugin** is a self-describing unit that registers one or more
RAGHub components on a
:class:`raghub.plugins.registry.PluginRegistry`. Plugins are the
supported way to swap implementations. The
:class:`raghub.RAG` facade accepts a `registry=` argument; the
default registry can be replaced wholesale or extended.

## What a plugin may register

| Helper | Type slot |
|---|---|
| `register_converter(name, converter)` | `DocumentConverter` |
| `register_chunker(name, chunker)` | `Chunker` |
| `register_embedder(name, embedder)` | `EmbeddingProvider` |
| `register_vector_store(name, store)` | `VectorStore` |
| `register_knowledge_repo(name, repo)` | `KnowledgeRepository` |
| `register_generator(name, generator)` | `Generator` |
| `register_structured(name, provider)` | `StructuredOutputProvider` |
| `register_telemetry(name, logger, metrics)` | `Logger`, `Metrics` (paired) |
| `register_evaluator(name, evaluator)` | `Evaluator` |
| `register_factory(name, factory)` | Generic callable |

Each type slot has a matching `get_*` resolver:

```python
registry = PluginRegistry()
registry.get_converter("marker")
registry.get_embedder("litellm")
registry.get_telemetry("langfuse")  # returns (Logger, Metrics)
```

## Authoring a plugin

A plugin is any object that exposes:

- `name: str` — a stable identifier.
- `version: str` — semantic version.
- `register(registry: PluginRegistry) -> None` — called once
  during registration; the plugin should use `registry.register_*`
  helpers.

```python
from raghub.plugins.registry import PluginRegistry
from raghub.interfaces.structured import StructuredOutputProvider


class MyStructuredProvider(StructuredOutputProvider):
    async def generate(self, *, response_model, question, context):
        return response_model(answer=42)


class MyPlugin:
    name = "my-plugin"
    version = "1.0.0"

    def register(self, registry: PluginRegistry) -> None:
        registry.register_structured("my-provider", MyStructuredProvider())


def my_plugin_factory():
    return MyPlugin()
```

You can swap any collaborator the same way — for example a custom
Markdown chunker:

```python
from raghub.interfaces.chunker import Chunker
from raghub.plugins.registry import PluginRegistry


class MyMarkdownChunker(Chunker):
    name = "my-markdown-chunker"

    def split(self, text: str) -> list[str]:
        return [chunk for chunk in text.split("\n\n") if chunk.strip()]


registry = PluginRegistry()
registry.register_chunker("markdown", MyMarkdownChunker())
```

## Registration via entry points

Add the plugin to your package's `pyproject.toml`:

```toml
[project.entry-points."raghub.plugins"]
my_plugin = "my_package.module:my_plugin_factory"
```

`PluginRegistry.discover_entrypoints()` uses
`importlib.metadata.entry_points(group=group)` to enumerate every
installed package's registered entry points under the given group
name. For each entry point it calls ``entry.load()`` (which imports
and returns the factory callable), invokes the factory, and then
calls ``plugin.register(registry)``:

```python
entries = importlib.metadata.entry_points(group="raghub.plugins")
for entry in entries:                     # e.g. "my_plugin"
    factory = entry.load()                # imports my_package.module:my_plugin_factory
    plugin = factory()                    # returns MyPlugin()
    plugin.register(registry)             # registers components
```

Usage from application code:

```python
loaded = PluginRegistry().discover_entrypoints(group="raghub.plugins")
print(f"loaded {loaded} plugins")
```

The RAGHub project itself ships with an empty entry-point group
by default; see `pyproject.toml`.

## Runtime registration

```python
from raghub import RAG
from raghub.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.register_structured("my-provider", MyStructuredProvider())

rag = RAG(registry=registry)
```

Pre-registering on a registry does not, by itself, replace the
collaborator on an existing facade instance — pass the replacement
to the constructor (e.g. `RAG(structured=registry.get_structured(...))`)
or reconstruct.

## Wiring every spec component

The full set of extension points mapped to the spec libraries:

| Spec component | Interface | Default implementation |
|---|---|---|
| Document conversion | `DocumentConverter` | `MarkerConverter` (Marker) → `PlainTextConverter` fallback |
| Chunking | `Chunker` | `ChonkieChunker` (Chonkie) → `WordWindowChunker` fallback |
| Embeddings | `EmbeddingProvider` | `LiteLLMEmbeddingProvider` → `HashingEmbeddingProvider` fallback |
| LLM | `LLMProvider` | `LiteLLMProvider` → `HeuristicLLMProvider` fallback |
| Vector store | `VectorStore` | `QdrantVectorStore` (when `QDRANT_URL` set) → `InMemoryVectorStore` fallback |
| Generator | `Generator` | `DefaultGenerator` wrapping the LLM |
| Reranker | `Reranker` | `IdentityReranker` |
| Structured output | `StructuredOutputProvider` | `InstructorStructuredOutputProvider`; `None` when unavailable |
| Telemetry | `TelemetryProvider` | `LangfuseTelemetryProvider` → `NoOpTelemetry` fallback (wrapped in `RedactingTelemetry`) |
| Conversation store | `ConversationStore` | `InMemoryConversationStore` |
| Knowledge repo | `KnowledgeRepository` | `InMemoryKnowledgeRepository` |
| Source manifest | `SourceManifest` | `./data/manifest.json` |
| Background ingestion | `BackgroundIngestionService` | `ResumableBackgroundIngestionService` (lazy) |

Every slot has an interface module under `raghub.interfaces/`. To
replace a collaborator, write a class implementing the interface
and pass it to the constructor.

## Testing a plugin

```python
def test_my_plugin():
    registry = PluginRegistry()
    MyPlugin().register(registry)
    assert "my-provider" in registry.structured
    assert registry.get_structured("my-provider").name == "my-provider"
```

The plugin system never modifies the default registry — tests are
free to instantiate fresh registries and register the plugin
without interfering with other tests.
