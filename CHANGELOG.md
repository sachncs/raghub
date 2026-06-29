# Changelog

All notable changes to RAGHub are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Public ``RAG`` facade** (`raghub.api.rag.RAG`) is the single
  recommended entry point. Every spec component — Marker,
  Chonkie, LiteLLM, Instructor, Qdrant, Langfuse — is wired as a
  default but can be replaced through the constructor.
- **Multi-user RBAC** at the retrieval layer via
  ``allowed_company_filter``. Admins see every company;
  non-admins see only chunks whose ``company`` is in their
  ``allowed_companies``. The LLM only ever receives authorised
  context. Unauthorised users with empty allow-lists see no
  documents.
- **Conversational RAG** with session-scoped history. Every
  query accepts ``session_id=``; the ``InMemoryConversationStore``
  prepends the most recent turns so the LLM can answer
  follow-up questions.
- **Streamlit UI** (`streamlit_app.py`) with ``st.chat_message``
  and ``st.chat_input``, citation rendering per turn, per-user
  document upload scoped to the user's company, and persistent
  sign-in. Pre-seeds five demo users.
- **Token-usage tracking**: ``LiteLLMProvider.generate`` and
  ``astream`` populate ``self.last_usage``; ``DefaultGenerator``
  exposes ``record_tokens()``; the ``QueryPipeline`` forwards
  tokens to the telemetry provider on every call.
- **Real streaming**: ``RAG.astream`` routes through
  ``QueryPipeline.stream`` → ``DefaultGenerator.astream`` →
  ``LiteLLMProvider.astream`` (with
  ``stream_options={"include_usage": True}``).
- **Incremental indexing** by SHA-256 content hash.
  ``IngestPipeline`` short-circuits when the file hasn't changed.
- **Resumable background ingestion** via
  ``ResumableBackgroundIngestionService`` (persistent job
  ledger in SQLite).
- **Retrieval-quality metrics** (``raghub.evaluation.metrics``):
  ``recall_at_k``, ``precision_at_k``, ``mean_reciprocal_rank``,
  ``context_recall``, ``context_precision``, ``faithfulness``,
  ``answer_correctness``. Integrated into ``FinanceBenchEvaluator``.
- **Performance benchmark** (``bench/benchmark.py``) measures
  startup time, ingestion throughput, query latency (p50/p95),
  queries-per-second under concurrency, and peak RSS.
- **10 spec libraries** wired as defaults: Marker (PDF → MD),
  Chonkie (chunking), LiteLLM (LLM + embeddings), Instructor
  (structured output), Qdrant (vector store), Langfuse
  (telemetry), OKF (knowledge), datasets (FinanceBench), pypdf
  (legacy), Tika-style parser fallback.
- **Open Knowledge Format (OKF)** is the canonical persisted
  representation. ``KnowledgeBundle`` round-trips through
  ``to_okf`` / ``from_okf``.
- **Plugin registry** with entry-point discovery
  (``[project.entry-points."raghub.plugins"]``).
- **Structured output** via Instructor; ``RAG.query(..., response_model=MyModel)``
  returns a typed ``BaseModel`` in ``Response.structured``.
- **Comprehensive test suite**: 315+ tests across 30+ test
  files, including 10-user concurrent multi-user tests, security
  tests, end-to-end tests, retrieval-metric tests, and benchmark
  tests.

### Changed

- **Public API surface** is now the new ``RAG`` facade. The legacy
  ``DynamicRagApplication`` and ``Build_application`` entry points
  remain available for backwards compatibility but are no longer
  the recommended path.
- **Configuration precedence**: env > TOML > YAML > defaults.
  ``AppSettings.override(**changes)`` provides runtime
  overrides.
- **Embedders and LLMs** fall back to in-process
  (``HashingEmbeddingProvider``, ``HeuristicLLMProvider``) when no
  API key is configured.
- **Default chunker** uses Chonkie when available, falls back to
  ``WordWindowChunker``.

### Security

- ``JWT_SECRET`` must be ≥ 32 bytes in production (PyJWT's
  ``InsecureKeyLengthWarning`` is now an error in CI).
- ``RedactingTelemetry`` scrubs secret-looking kwargs before
  forwarding to the telemetry backend.
- Per-user retrieval isolation: unauthorised users receive empty
  result sets, not 403 errors.

### Fixed

- ``RAG.astream`` is now a real token stream (previously
  materialised the full answer before yielding).
- ``LiteLLMProvider.astream`` populates ``last_usage`` with
  ``stream_options={"include_usage": True}``.
- ``MarkerConverter.convert`` raises a clear
  ``ConfigurationError`` on empty / non-PDF input instead of a
  raw PDFium crash.
- ``RAG.ingest`` and ``RAG.aingest`` raise ``RagHubError`` on
  empty bytes.
- The 4 ``InsecureKeyLengthWarning`` instances that fired in
  every test run are gone.
- Fixed circular imports between ``raghub.domain`` and
  ``raghub.repositories`` (legacy code now lazy-imports).

### Deferred

- ``AppSettings.jwt_secret`` will move to ``SecretStr`` in a future
  release. The current ``str`` annotation is required for the
  Pydantic 1.x-style dataclass shape; migrating to ``BaseModel``
  is a larger refactor.
- The 48 remaining ``mypy`` warnings are concentrated in the
  legacy ``services/``, ``repositories/``, and ``domain/`` modules.
  They will be removed when the legacy code is removed in a future
  release.

### Removed

- **Dead code**: ``raghub/cache/``, ``raghub/memory/``,
  ``raghub/monitoring/`` (placeholder packages), the legacy
  ``raghub.llm.nvidia`` and ``raghub.embeddings.nvidia`` modules
  (superseded by LiteLLM), the duplicate
  ``raghub/vectorstores/`` (plural) package (merged into
  ``raghub/vectorstore/``), and the ``TemplatePromptBuilder``
  class.
- 104 stale ``,cover`` files (replaced by ``,cover`` in
  ``.gitignore``).
- The legacy CLI's ``login`` command (replaced by the FastAPI
  ``/auth/login`` endpoint and the demo users in the Streamlit
  UI).
- Standalone ``evaluate_financebench.py`` at the repo root
  (replaced by ``raghub-financebench`` console script and
  ``raghub.cli.eval_cmd``).

## [1.0.0] - 2024-Q4

### Added

- Initial release with the legacy multi-tenant service stack
  (DynamicRagApplication, passwordless login, ZVec / NVIDIA LLM).
- 78+ tests, 76% coverage, 0 ruff errors, 0 mypy errors.
