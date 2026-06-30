# Development Guide

This is for people hacking on RAGHub itself. App developers should
read [`getting-started.md`](getting-started.md) instead.

## Environment setup

```bash
git clone https://github.com/sachn-cs/raghub.git
cd raghub

./setup.sh                       # creates .venv, installs dev extras
source .venv/bin/activate

pip install -e ".[api,ui,dev]"   # for a fresh checkout
```

`setup.sh` provisions a `.venv`, installs the project in editable
mode with the `api`, `ui`, and `dev` extras, and prints the next
steps. `./cleanup.sh` removes `.venv`, `data/`, and
`__pycache__/` — `./setup.sh` rebuilds from scratch.

## Running tests

```bash
pytest tests/ -v                 # all tests
pytest tests/ -x                 # stop on first failure
pytest tests/ -xvs               # verbose, stop on first failure

# Single test (full traceback on failure)
pytest tests/test_platform.py::test_ingest_and_query_isolated_access -xvs
```

The test suite covers the spec libraries, multi-user RBAC, the
conversational pipeline, structured output, telemetry scrubbing,
the resumable ingestion service, retrieval metrics, and the
performance benchmark. There are 315+ tests across 30+ files.

### FinanceBench

```bash
# pytest path — controlled by env var, kept off by default
FINANCEBENCH_EVAL=1 pytest tests/test_financebench.py -xvs
```

Or via the bundled CLI / console script:

```bash
raghub eval financebench --examples 25
raghub-financebench --examples 25
```

The evaluator reports `recall@K`, `precision@K`, MRR, Faithfulness,
Context Recall, Context Precision, and Answer Correctness in
addition to the binary pass-rate.

### Performance benchmark

```bash
python -m bench.benchmark --documents 100 --queries 200 --concurrency 8
```

Measures startup time, ingestion throughput, query latency (p50
and p95), queries-per-second under concurrency, and peak RSS. The
report is written to `bench/report.json`. The script is also
exposed as `raghub-benchmark`.

## Code style

- **Google Python Style Guide** for docstrings.
- **Type hints** on every public function and method (project is
  `mypy` clean on `raghub/` proper).
- **No `_` prefix** on public names; use the explicit module-level
  `__all__` to declare the public surface.
- **Active-record domain pattern** for the legacy
  `DynamicRagApplication`; the new `RAG` facade uses composable
  pipelines instead.
- **Spec libraries** are first-class imports; the framework wires
  Marker, Chonkie, LiteLLM, Instructor, Qdrant, and Langfuse as
  defaults and falls back to in-process providers when the
  libraries are missing.

## Linting and type checking

```bash
ruff check raghub/ tests/
mypy raghub/
```

The project is clean: zero ruff errors in `raghub/`. There are
48 remaining mypy warnings concentrated in the legacy `services/`,
`repositories/`, and `domain/` modules — these are slated to be
removed when the legacy code is replaced (see
[`future.md`](../future.md)).

## Adding a new component

The cleanest way is a **plugin**:

1. Create the implementation under your own package
   (`my_plugin/`). Implement one or more of the protocols in
   `raghub.interfaces/`.
2. Expose a factory in `pyproject.toml`:

   ```toml
   [project.entry-points."raghub.plugins"]
   my_plugin = "my_plugin.module:my_plugin_factory"
   ```

3. In the factory, return an object with `name`, `version`, and
   `register(registry)`:

   ```python
   class MyPlugin:
       name = "my-plugin"
       version = "0.1.0"

       def register(self, registry):
           registry.register_chunker("my-chunker", MyChunker())
   ```

4. Plug the registry into the facade:

   ```python
   from raghub import RAG
   from raghub.plugins.registry import PluginRegistry

   rag = RAG(registry=PluginRegistry().discover_entrypoints())
   # or a single component
   rag = RAG(chunker=registry.get_chunker("my-chunker"))
   ```

See [`plugins.md`](../plugins.md) for the full set of extension
points (converters, chunkers, embedders, vector stores,
generators, structured-output, telemetry, evaluators).

## Adding a new conversion target

The `RAG` facade's default chain is `MarkerConverter → PlainTextConverter`.
Plugins register additional converters with `registry.register_converter(name, converter)`.

To support a new file format end-to-end:

1. Add a converter at `raghub/converters/<format>.py`. It must
   implement the `DocumentConverter` interface (returns a
   `KnowledgeBundle`).
2. If it requires an external tool, add the dependency to
   `pyproject.toml`'s `dependencies` (or as an optional extra).
3. Either wire it into `default_converter()`'s fallback chain or
   publish as a plugin.

## Adding a new embedding / LLM provider

Embedding and LLM providers reach the facade through the
`EmbeddingProvider` and `LLMProvider` interfaces in
`raghub.interfaces`. The two easiest paths:

1. **Use LiteLLM** — any new provider LiteLLM supports works
   out of the box. Configure it with
   `LiteLLMProvider(model="<provider>/<model>")` and pass to the
   constructor.
2. **Write an adapter** — implement the interface, optionally
   register it as a plugin, and pass to the facade constructor.

## Adding a new vector store

Implement `raghub.interfaces.vectorstore.VectorStore`. The
simplest starter is `raghub.vectorstore.memory.InMemoryVectorStore`
— copy that. To activate for the default facade, extend
`default_vector_store()` or publish as a plugin.

## Resetting the environment

```bash
./cleanup.sh      # removes .venv, data/, __pycache__/
./setup.sh        # rebuild from scratch
```

To blow away just the data directory:

```bash
rm -rf data/
```

This drops the manifest, the ingestion job ledger, registry, and
session store. The next ingest runs as a cold start.

## Contributing

- Match the surrounding code's comment density and naming.
  When adding a new module, write the public docstring in the
  style of `raghub.api.rag.RAG`.
- New public API goes through `RAG(...)` as the integration
  point; legacy paths remain reachable for backward
  compatibility.
- Add tests under `tests/` (group by component, e.g.
  `tests/test_<area>_<feature>.py`).
- Update [`../CHANGELOG.md`](../../CHANGELOG.md) and the related
  reference doc under `docs/reference/` or `docs/guide/` in the
  same change.
- Run `pytest tests/ -x` and `ruff check raghub/ tests/`
  locally before pushing.
