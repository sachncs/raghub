# Tier 5 — CLI for v0.7.5 / v0.7.6 / v0.7.8 (Items 25-30)

The v0.7.5/6/8 plans shipped adapters but no CLI to operate them.
Tier 5 ships the CLI commands the plan promised.

---

## Item 25 — CLI `raghub migrate pgvector`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: New `MigratePgVectorCommand` invoking `PgVectorStore(dsn, embedding_dim=settings.embedding_dim).initialize()`. Flags: `--dsn` (required), `--vector-dim` (default from settings).
- **Test**: `tests/test_cli.py::test_migrate_pgvector_runs` (mocked).
- **Acceptance criteria**:
  - R3 — `MigratePgVectorCommand` is single-word.
  - T3 — pass.
- **Success criteria**:
  - `raghub migrate pgvector --dsn "postgres://..." --vector-dim 384` exits 0 and calls `PgVectorStore.initialize()`.
  - Without `--dsn`, the command exits 1 with a clear error.

---

## Item 26 — CLI `raghub tenant list | create | delete`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: Three sub-commands on `TenantCommand`:
  - `list` — iterate the on-disk `TenantRegistry`.
  - `create <id>` — register the tenant.
  - `delete <id>` — unregister; refuses if data exists unless `--force`.
- **Test**: `tests/test_cli.py::test_tenant_list_create_delete_round_trip`.
- **Acceptance criteria**:
  - T3 — pass.
  - R9 — assertion on tenant registry state.
- **Success criteria**:
  - `raghub tenant create acme` succeeds.
  - `raghub tenant list` shows `acme`.
  - `raghub tenant delete acme` succeeds; default refuses with data, `--force` succeeds.

---

## Item 27 — CLI `raghub migrate tenant-split`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: Single sub-command wrapping `migrate_tenant_split`. Flags: `--from`, `--to`, `--source-dsn`, `--target-dsn`, `--tenant-id`.
- **Test**: `tests/test_cli.py::test_migrate_tenant_split_runs` (mocked).
- **Acceptance criteria**:
  - T3 — pass.
  - Direction values are validated (`row_level`, `schema_per_tenant`, `database_per_tenant`).
- **Success criteria**:
  - `raghub migrate tenant-split --from row_level --to schema_per_tenant --source-dsn ... --target-dsn ...` calls `migrate_tenant_split(...)` with the parsed arguments.
  - Invalid `--from` value exits 1 with a clear error.

---

## Item 28 — CLI `raghub backup`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: `raghub backup --output <path.tar.zst> [--tenant <id>]` calls `create_snapshot(root, ...)` + `write_archive(manifest, files, output_path)`.
- **Test**: `tests/test_cli.py::test_backup_creates_archive`.
- **Acceptance criteria**:
  - R3 — `BackupCommand` is single-word.
  - T3 — pass.
  - O3 — format version pinned in archive.
- **Success criteria**:
  - `raghub backup --output /tmp/backup.tar.zst` produces a file.
  - `raghub backup verify --input /tmp/backup.tar.zst` (item 30) passes.

---

## Item 29 — CLI `raghub restore`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: `raghub restore --input <path.tar.zst> [--target-dir <path>]` calls `restore_snapshot(archive_path, target_dir, ...)`.
- **Test**: `tests/test_cli.py::test_restore_round_trip`.
- **Acceptance criteria**:
  - R6 — manifest signature verified before any restore.
  - T3 — pass.
- **Success criteria**:
  - After backup → restore on a fresh target dir, files match SHA-256.
  - A tampered archive causes exit 1 (no partial restore).

---

## Item 30 — CLI `raghub backup verify`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: `raghub backup verify --input <path.tar.zst>` calls `verify_archive(archive_path)`. Exit code 0 on success, 1 on tamper.
- **Test**: `tests/test_cli.py::test_backup_verify_succeeds_on_valid_archive`, `::test_backup_verify_fails_on_tampered_archive`.
- **Acceptance criteria**:
  - T3 — pass.
  - HMAC signature verified.
- **Success criteria**:
  - `raghub backup verify --input <valid>` exits 0.
  - `raghub backup verify --input <tampered>` exits 1.

---

## Tier 5 acceptance gate

- `raghub migrate pgvector --help` shows the help text.
- `raghub tenant create foo` persists a tenant record.
- `raghub backup → raghub restore` is a round-trip without data loss.
- `raghub backup verify` rejects a tampered archive.
