# Plugin Development

A **plugin** is a self-describing unit that registers one or more
RAGHub components (converters, chunkers, embedders, vector stores,
retrievers, rerankers, generators, telemetry providers, evaluators)
on a :class:`raghub.plugins.registry.PluginRegistry`.

Plugins are the supported way to swap implementations. The
:class:`raghub.RAG` facade accepts a `registry=` argument, and the
default registry can be replaced wholesale or extended.

## Authoring a Plugin

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

## Registration via Entry Points

Add the plugin to your package's `pyproject.toml`:

```toml
[project.entry-points."raghub.plugins"]
my_plugin = "my_package.module:my_plugin_factory"
```

`PluginRegistry.discover_entrypoints()` will instantiate the factory
and call `register(registry)` automatically.

## Runtime Registration

```python
from raghub import RAG
from raghub.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.register_structured("my-provider", MyStructuredProvider())

rag = RAG(registry=registry)
```

## Discovering Plugins in Code

```python
registry = PluginRegistry()
loaded = registry.discover_entrypoints(group="raghub.plugins")
print(f"loaded {loaded} plugins")
```

## Plugin Interface

A plugin is any object that exposes:

- `name: str` — a stable identifier.
- `version: str` — semantic version.
- `register(registry: PluginRegistry) -> None` — called once during
  registration; the plugin should use `registry.register_*` helpers.

## What a Plugin May Register

- `register_converter(name, converter)` — document converter
- `register_chunker(name, chunker)` — chunker
- `register_embedder(name, embedder)` — embedding provider
- `register_vector_store(name, store)` — vector store
- `register_knowledge_repo(name, repo)` — knowledge repository
- `register_generator(name, generator)` — answer generator
- `register_structured(name, provider)` — structured-output provider
- `register_telemetry(name, logger, metrics)` — telemetry pair
- `register_evaluator(name, evaluator)` — benchmark evaluator
- `register_factory(name, factory)` — generic factory

## Testing a Plugin

```python
def test_my_plugin():
    registry = PluginRegistry()
    MyPlugin().register(registry)
    assert "my-provider" in registry.structured
```
