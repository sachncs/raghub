# archive/ — raghub 0.9.x (Python, deprecated)

This directory holds the final Python release of raghub (0.9.x). It is
**frozen, read-only, and unsupported**. No further Python releases will
ship. Do not open issues against this tree.

**The active codebase is the TypeScript monorepo at the repository root.**
See [`../README.md`](../README.md) and [`../AGENTS.md`](../AGENTS.md).

## What to do if you arrived here

| You want to… | Use… |
|---|---|
| Build a new project | `npm install @raghub/core @raghub/orchestrator @raghub/api` |
| Run the CLI | `npx @raghub/cli init` |
| Talk to the API | `cd apps/web && pnpm dev` |
| Migrate data out of 0.9.x | `npx @raghub/cli migrate pgvector-to-sqlite --dsn <pg>` then `npx @raghub/cli ingest import` |

## What lives here

- `raghub/` — the 0.9.x Python package, untouched.
- `tests/` — the 0.9.x Python test suite.
- `pyproject.toml` — the 0.9.x Python build.
- `docs/` — the 0.9.x MkDocs reference.
- `data/` — development SQLite databases from 0.9.x testing (not for
  production use).
- `devtools/`, `lint/`, `reports/`, `todo/`, `setup.sh`, `cleanup.sh`,
  `Makefile`, `mkdocs.yml`, `RELEASE_NOTES.md`, `CHANGELOG.md`,
  `fix.md`, `.pre-commit-config.yaml` — historical Python tooling and
  artefacts.

## Deprecation shim

`raghub/__deprecated__.py` is a single-purpose import shim. If you
_somehow_ still depend on the Python package, import it to print a
deprecation warning at startup:

```python
import raghub.__deprecated__  # noqa: F401
```

## License

MIT, unchanged. See [`../LICENSE`](../LICENSE).