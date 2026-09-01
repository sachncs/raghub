# CLI reference

`apps/cli` ships the `revex` binary, built on [Commander](https://github.com/tj/commander.js).
Run via:

```bash
pnpm --filter @revex/cli dev -- <cmd> [options]
# or after a build:
pnpm --filter @revex/cli run cli -- <cmd> [options]
```

## Global options

| Option | Description |
|---|---|
| `-v, --version` | Print version. |
| `-h, --help` | Show help. |

Environment: `REVEX_API_BASE` (default `http://localhost:3000`) is the base URL
for every HTTP subcommand. The CLI reads `REVEX_JWT_SECRET` /
`REVEX_EMBEDDER_API_KEY` only when it runs in-process (e.g. `server`,
`eval`).

## Subcommands

| Command | Purpose |
|---|---|
| [init](#revex-init) | Initialize a workspace. |
| [server](#revex-server) | Start the API server. |
| [ingest](#revex-ingest) | Ingest files into a workspace. |
| [query](#revex-query) | Ask a question. |
| [config](#revex-config) | Get/set config values. |
| [tenant](#revex-tenant) | List / inspect tenants. |
| [backup](#revex-backup) | Snapshot a workspace. |
| [queue](#revex-queue) | Inspect / drive the job queue. |
| [migrate](#revex-migrate) | Run migrations. |
| [feedback](#revex-feedback) | Read feedback. |
| [eval](#revex-eval) | Run a benchmark. |

---

### `revex init`

Initialize a new workspace.

```
revex init [options] -n <name>
```

| Option | Description |
|---|---|
| `-n, --name <name>` | Workspace name (required). |
| `--path <path>` | Workspace DB path. |
| `--passphrase <pass>` | Encryption passphrase (min 8 chars). |
| `--admin-email <email>` | Admin email. |
| `--admin-password <pass>` | Admin password (min 8 chars). |
| `--llm-provider <p>` | LLM provider. |
| `--llm-model <m>` | LLM model (default `gpt-4.1`). |
| `--llm-api-key <k>` | LLM API key. |
| `--pass` | Prompt for interactively-specified values. |

### `revex server`

Start the HTTP API server.

```
revex server [options]
```

| Option | Description |
|---|---|
| `--port <port>` | Listen port (default `3000`). |
| `--workspace-dir <dir>` | Workspace data directory. |
| `--pass` | Prompt for the passphrase. |

### `revex ingest`

Ingest files into a workspace over HTTP.

```
revex ingest [options] <doc...>
```

| Option | Description |
|---|---|
| `<doc...>` | File paths (required). |
| `-c, --collection-id <id>` | Collection / category. |
| `--api-base <url>` | API base URL (default `REVEX_API_BASE`). |

### `revex query`

Ask a question and print the answer.

```
revex query [options] <question>
```

| Option | Description |
|---|---|
| `<question>` | The question (required). |
| `--api-base <url>` | API base URL. |

### `revex config`

Get or set configuration values.

```
revex config [options]          # list all
revex config get <key>          # read one
revex config set <key> <value>  # write one
```

### `revex tenant`

Inspect tenants via the registry.

```
revex tenant [options]                 # list
revex tenant describe <id>             # show one tenant
```

### `revex backup`

Snapshot a workspace.

```
revex backup [options] <workspaceId>
```

| Option | Description |
|---|---|
| `<workspaceId>` | Workspace to snapshot (required). |
| `--out <dir>` | Output directory (default `snapshots/`). |

### `revex queue`

Inspect / drive the job queue.

```
revex queue list [options]    # list jobs
revex queue stats             # queue statistics
revex queue purge             # clear the queue
revex queue submit <kind>     # enqueue a job by kind
```

| Option (list) | Description |
|---|---|
| `--status <s>` | Filter by `pending | running | done | failed`. |
| `--limit <n>` | Max rows (default 50). |

### `revex migrate`

Run database migrations.

```
revex migrate [options]
```

In-process: uses the configured workspace path + passphrase.

### `revex feedback`

Read feedback records.

```
revex feedback [options]
```

| Option | Description |
|---|---|
| `--workspace-id <id>` | Workspace to query. |
| `--limit <n>` | Max records (default 20). |

### `revex eval`

Run a retrieval benchmark.

```
revex eval [options] <benchmark>
```

| Argument | Value | Description |
|---|---|---|
| `<benchmark>` | `finance` | Finance benchmark metrics run. |
| | `frames` | FRAMES benchmark run. |
| | `lost-in-middle` | Lost-in-the-middle probe (placeholder). |

| Option | Description |
|---|---|
| `-i, --input <jsonl>` | JSONL input of `QASample[]`. |
| `--output <json>` | Write aggregate metrics to this file. |
| `--limit <n>` | Cap the number of samples. |
| `--list` | List available benchmarks. |

The CLI uses a stub retrieval/LLM, so runs are functional but not faithful to
a live embedder/provider.