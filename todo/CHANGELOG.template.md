# CHANGELOG entry template

Every release uses this exact skeleton. The release file contains a
pre-filled block; copy that block into the matching `## [<VERSION>]`
section at the top of `CHANGELOG.md` as the final acceptance step.

```markdown
## [<VERSION>] - <YYYY-MM-DD>

### Changed (BREAKING — R4)

Per `AGENTS.md` rule R4 and the project policy, this release introduces
breaking changes without aliases, shims, or a deprecation period. Old
names simply do not exist after this release. There is no migration
path by design — update your code to the new names.

### Added

- <bullet>

### Changed

- <bullet>

### Removed

- <bullet>

### Fixed

- <bullet>

### Security

- <bullet>

### Notes

- <bullet or omit the section>
```

## How to fill it in

- **Added**: net-new public symbols or features.
- **Changed**: behaviour changes to existing public symbols, even if
  the name is preserved.
- **Removed**: deletions of public symbols, endpoints, dependencies.
- **Fixed**: correctness bug fixes. Reference the bug with a short
  description; full detail lives in the release file.
- **Security**: anything security-relevant — removed attack surface,
  new mitigations, threat models committed.
- **Notes**: optional. Use only for things that don't fit the
  categories above (e.g. moved docs, bumped internal versions).

## Style rules

- Bullets are imperative-mood sentences ending with a period.
  - Good: `- raghub.Prometheus re-export removed.`
  - Bad: `- Removed raghub.Prometheus.`
- One concept per bullet. If you need "and" twice, split into two
  bullets.
- Reference file paths in backticks when relevant:
  `- \`raghub/telemetry.py\`: PrometheusMetrics class deleted.`
- Do **not** link to PRs from CHANGELOG. PR links belong in release
  notes or the release commit message.
- Do **not** include upgrade instructions. R4 forbids them.

## Example: filled entry (from v0.7.0)

```markdown
## [0.7.0] - 2026-08-15

### Changed (BREAKING — R4)

Per `AGENTS.md` rule R4, no aliases or shims. Old names simply do
not exist after this release.

### Added

- Langfuse telemetry provider promoted to a core dependency; no
  extra install needed.

### Changed

- `raghub.telemetry`: metric emission now flows through
  `langfuse.score(...)`; silent no-op when Langfuse unconfigured.
- `record_rerank_latency` and `record_long_context` semantics now
  use Langfuse scores instead of Prometheus histograms.

### Removed

- `raghub.Prometheus`, `raghub.PrometheusMetrics`, `raghub.NullMetrics`,
  `raghub.DEFAULT_METRICS_REGISTRY`, `raghub.set_active_metrics`,
  `raghub.record_rerank_latency`, `raghub.record_long_context`,
  `raghub.generate_latest`, and the `/metrics` FastAPI route deleted.
- `prometheus-client` dependency removed from `pyproject.toml` and
  from the transitive dependency closure.
- 17 dedicated Prometheus test cases deleted.

### Fixed

- `Instructor.astream` returned `None`; now returns the inner stream.
- `DefaultGenerator.generate` signature reconciled with docstring.
- `ChunkRef.__init__` self-referential annotation fixed.
- `UnitOfWork(...)` call signature reconciled with class signature.
- `GraphIndex.delete_for_document` compared the wrong id field.
- `Raptor.add_chunks` replaced the levels list; now appends and
  dedupes by id.
- `financebench --examples 0` evaluated zero rows; now loads all.
- `record_rerank_latency` histogram was missing the `provider` label.
- `record_long_context` accepted `seconds` and dropped it.
- `RedactingTelemetry.redact_record` now recurses into nested dicts.
- Dead `LITELLM_AVAILABLE` branch deleted.
- `LiteLLM.async_generate` translates `asyncio.TimeoutError` to
  `GenerationError`.
- `SqliteStore.hybrid_search` renamed to `search_hybrid`; raises
  `ConfigurationError` when BM25 is unavailable.
- `DocStore.try_insert` now honours `max_retries` with exponential
  backoff.
- `UnitOfWork.__init__` no longer uses `assert` for runtime validation.
- `# type: ignore` in `raghub.conv` removed; call sites fixed.
- `raghub.eval.evaluate` initialises `contexts` and `retrieved_ids`
  to `None` and guards correctly.
- `Agent.__generate_reply` re-raises generation errors as
  `GenerationError`, not `AgentBudgetError`.
- `validate_cors` raises `ConfigurationError`, not `RuntimeError`.
- `services/__init__.py` `__all__` deduplicated (`Facade`,
  `Document`).
- `models.py` `__all__` deduplicated (`Chunk`, `Document`).
- `stores/__init__.py` extracted `__serialize_overrides()` helper.
- `Agent.iterate` decomposed into `__check_budget`,
  `__build_prompt`, `__dispatch`, `__render_final`.
- `RouteGroup` split into `HealthRouter`, `AuthRouter`,
  `DocumentRouter`, `QueryRouter`, `AgentRouter`, `AdminRouter`;
  internal `_`-prefix methods renamed to public or `__dunder__`.

### Security

- `/metrics` route removed; no internal counter leakage over HTTP.
- `prometheus-client` transitive dependency closure removed.

### Notes

- Observability is now Langfuse-only. Custom metric backends must be
  wired via the Langfuse scores API or via community telemetry
  adapters registered through `PluginRegistry`.
```
