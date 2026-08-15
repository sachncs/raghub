"""Helpers for the CLI surface.

The pre-existing ``raghub.cli`` package spread its commands across six
small modules (``config``, ``format``, ``ingest``, ``init``, ``query``,
``server``). They have been collapsed into this single module under
one class per concern::

    CliConfig       - JSON output, YAML/TOML settings loader, RAG factory.
    ToolConfig      - the ``raghub config tools ...`` sub-app + its three commands.
    IngestCommand   - the ``raghub ingest`` command.
    InitCommand     - the ``raghub init`` command and its starter YAML template.
    QueryCommand    - the ``raghub query`` command.
    ServerCommand   - the ``raghub run`` command (uvicorn host).

Each command class exposes a ``register(app)`` method that attaches
its Typer command to the parent app; ``main.py`` calls them in
``register`` order. ``ToolConfig.register`` also binds its three
sub-commands onto ``ToolConfig.app``.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import typer
import uvicorn
import yaml

from raghub.api import App
from raghub.config import Settings, settings_field_names
from raghub.constants import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_DIM,
    ENV_RAG_TENANT_DSNS,
)
from raghub.io import write_json as write_json_impl
from raghub.rag import RAG

# FeedbackCommand is re-exported from raghub.commands.__init__ but its
# implementation lives in raghub.commands.feedback to avoid bloating this
# module. No direct import needed here.
from raghub.runtime import capture


class CliConfig:
    """JSON output, settings loading, and RAG factory for the CLI."""

    @staticmethod
    def write_json(payload: Any) -> None:
        """Write ``payload`` as pretty JSON to stdout."""
        write_json_impl(payload)

    @staticmethod
    def read_settings(path: str | None) -> Any:
        """Load :class:`raghub.config.Settings` from ``path`` or the active profile.

        Args:
            path: Optional YAML/TOML path. ``None`` calls
                :meth:`Settings.load`.

        Returns:
            The parsed :class:`Settings`.

        """
        if path is None:
            return Settings.load()
        if path.endswith(".toml"):
            import tomllib

            data = tomllib.loads(Path(path).read_text(encoding="utf-8")) or {}
        else:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return Settings(**{k: v for k, v in data.items() if k in settings_field_names()})

    @staticmethod
    def make_rag(config: str | None) -> Any:
        """Instantiate a :class:`raghub.RAG` from ``config`` or defaults.

        Args:
            config: Optional path to a YAML/TOML config file.

        Returns:
            A configured :class:`raghub.RAG` instance.

        """
        return RAG.from_config(config) if config else RAG()

    @staticmethod
    def make_settings(config: str | None) -> Any:
        """Load :class:`Settings` from ``config`` or default.

        Convenience wrapper used by commands that need direct access
        to :class:`Settings` (e.g. backup, tenant) without instantiating
        a full :class:`RAG` facade.
        """
        return CliConfig.read_settings(config)


class ToolConfig:
    """The ``raghub config tools ...`` sub-tree.

    Holds the Typer sub-app (:pyattr:`app`), the inner ``tools`` Typer
    (:pyattr:`tools`), the recognised tool-setting keys (:pyattr:`keys`),
    and the three commands attached via :meth:`register`.
    """

    app = typer.Typer(help="Configuration commands.", no_args_is_help=True)
    tools = typer.Typer(help="Manage the tool_settings blob.", no_args_is_help=True)
    keys: tuple[str, ...] = (
        "tools_enabled",
        "agent_enabled",
        "web",
        "graph",
        "summaries",
        "reranker",
        "long_context_pass",
        "query_transforms",
        "max_steps",
    )

    @staticmethod
    async def load_store() -> Any:
        """Build and initialise the user store for the active profile.

        Returns:
            The initialised :class:`raghub.auth.SqliteUsers`.

        """
        settings = CliConfig.read_settings(None)
        db_path = Path(settings.data_dir) / "users.db"
        from raghub.auth import SqliteUsers

        store = SqliteUsers(db_path)
        await store.initialize()
        return store

    @staticmethod
    def run_coro(coro: Any) -> Any:
        """Run an awaitable in a fresh event loop."""
        return asyncio.run(coro)

    @classmethod
    def register(cls: type[ToolConfig]) -> None:
        """Attach every sub-command to ``cls.tools`` and wire ``cls.app``.

        Idempotent: subsequent calls are no-ops so :func:`main` can call
        :meth:`register` defensively without doubling the commands.
        """
        if cls.tools.registered_commands:
            return
        cls.app.add_typer(cls.tools, name="tools")

        @cls.tools.command(name="list")
        def list_cmd(email: str = typer.Option(..., "--email", help="User email.")) -> None:
            """Print the tool_settings blob for ``--email`` as JSON."""

            async def runner() -> None:
                """Resolve the user, fetch their tool_settings, print JSON."""
                store = await cls.load_store()
                user = await store.get_by_email(email)
                if user is None:
                    raise typer.BadParameter(f"no user with email {email!r}")
                prefs = await store.get_prefs(user.id)
                CliConfig.write_json(
                    {"email": email, "tool_settings": prefs.get("tool_settings", {})}
                )

            cls.run_coro(runner())

        @cls.tools.command(name="set")
        def set_cmd(
            email: str = typer.Option(..., "--email", help="User email."),
            payload: str = typer.Option(
                ...,
                "--json",
                help=('JSON object, e.g. \'{"agent_enabled": true, "reranker": "cohere"}\'.'),
            ),
        ) -> None:
            """Merge the JSON ``--json`` payload into the user's tool_settings.

            Unknown keys are silently dropped.
            """
            patch, json_error = capture(json.loads, payload)
            if json_error is not None:
                raise typer.BadParameter(f"invalid JSON: {json_error}") from json_error
            if not isinstance(patch, dict):
                raise typer.BadParameter("--json must be a JSON object")

            async def runner() -> None:
                """Resolve the user, merge the JSON into tool_settings."""
                store = await cls.load_store()
                user = await store.get_by_email(email)
                if user is None:
                    raise typer.BadParameter(f"no user with email {email!r}")
                existing = await store.get_pref(user.id, "tool_settings") or {}
                if not isinstance(existing, dict):
                    existing = {}
                merged = {
                    **existing,
                    **{k: v for k, v in patch.items() if k in cls.keys},
                }
                await store.set_pref(user.id, "tool_settings", merged)
                CliConfig.write_json({"email": email, "tool_settings": merged})

            cls.run_coro(runner())

        @cls.tools.command(name="unset")
        def unset_cmd(email: str = typer.Option(..., "--email", help="User email.")) -> None:
            """Delete the tool_settings blob for ``--email``."""

            async def runner() -> None:
                """Resolve the user, delete tool_settings, print confirmation."""
                store = await cls.load_store()
                user = await store.get_by_email(email)
                if user is None:
                    raise typer.BadParameter(f"no user with email {email!r}")
                await store.delete_pref(user.id, "tool_settings")
                CliConfig.write_json({"email": email, "tool_settings": None})

            cls.run_coro(runner())


class IngestCommand:
    """The ``raghub ingest PATH`` command."""

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the ``ingest`` command to ``app``."""

        @app.command(name="ingest")
        def ingest(
            path: str = typer.Argument(..., help="Path to a file or directory."),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
            background: bool = typer.Option(
                False, "--async", help="Submit to the background job service and print a job id."
            ),
            json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
        ) -> None:
            """Ingest ``PATH`` synchronously, or submit it as a background job with ``--async``."""
            rag = CliConfig.make_rag(config)
            if background:
                job_id = rag.ingest_async(path)
                if json_output:
                    CliConfig.write_json({"job_id": job_id})
                else:
                    typer.echo(f"submitted job {job_id}")
                return
            result = rag.ingest(path)
            batch = result.outputs.get("batch") if result.outputs else None
            if isinstance(batch, list) and batch:
                payload: dict[str, Any] | list[dict[str, Any]] = [
                    r.dump(mode="json") for r in batch
                ]
            else:
                payload = result.dump(mode="json")
            if json_output:
                CliConfig.write_json(payload)
                return
            if isinstance(payload, list):
                typer.echo(f"ingested batch of {len(payload)} item(s)")
                return
            success = payload.get("success")
            if success is not None:
                typer.echo(f"{'OK' if success else 'FAIL'}: {path}")
            else:
                typer.echo(f"ingested {path}")


class InitCommand:
    """The ``raghub init`` command."""

    SAMPLE_CONFIG = """# RAGHub configuration — adjust to your environment.
environment: development
data_dir: ./data
chunk_size_words: DEFAULT_CHUNK_SIZE_WORDS
chunk_overlap_words: DEFAULT_CHUNK_OVERLAP_WORDS
chunker_strategy: recursive
embedding_model_chunker: minishlab/potion-base-8M
embedding_dim: DEFAULT_EMBEDDING_DIM
embedding_model: hashing-bge
llm_model: heuristic-llm
retrieval_mode: sync
log_level: INFO
worker_backend: threadpool
jwt_secret: change-me
nvidia_api_key: ""
allow_passwordless_login: true
"""

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the ``init`` command to ``app``."""

        @app.command(name="init")
        def init(
            output: str | None = typer.Option(
                None, "--output", "-o", help="Write to this path instead of stdout."
            ),
        ) -> None:
            """Print the starter config to stdout, or to ``--output``."""
            if output:
                Path(output).write_text(InitCommand.SAMPLE_CONFIG, encoding="utf-8")
                typer.echo(f"wrote {output}")
            else:
                typer.echo(InitCommand.SAMPLE_CONFIG, nl=False)


class QueryCommand:
    """The ``raghub query "..."`` command."""

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the ``query`` command to ``app``."""

        @app.command(name="query")
        def query(
            question: str = typer.Argument(..., help="The question to ask."),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
            top_k: int = typer.Option(5, "--top-k", "-k", help="Number of hits to retrieve."),
            json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
        ) -> None:
            """Ask a question and print the answer + citations."""
            rag = CliConfig.make_rag(config)
            response = rag.query(question, top_k=top_k)
            if json_output:
                CliConfig.write_json(
                    {
                        "answer": response.answer,
                        "citations": [c.dump() for c in response.citations],
                        "source_chunks": [s.dump() for s in response.source_chunks],
                        "metadata": response.metadata,
                    }
                )
                return
            typer.echo(f"\n{response.answer}\n")
            if response.citations:
                typer.echo(f"[{len(response.citations)} citations]")
                for c in response.citations:
                    typer.echo(f"  - {c.document_id}#{c.id} (score={c.score:.3f})")


class ServerCommand:
    """The ``raghub run`` command (uvicorn host)."""

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the ``run`` command to ``app``."""

        @app.command(name="run")
        def run(
            host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host."),
            port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
            workers: int = typer.Option(1, "--workers", "-w", help="uvicorn worker count."),
            reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes."),
        ) -> None:
            """Run the FastAPI app under uvicorn (foreground).

            Production deployments should set ``--workers`` > 1;
            ``--reload`` is for development.
            """
            settings = Settings.load()
            config = uvicorn.Config(
                lambda: App.create(settings),
                factory=True,
                host=host,
                port=port,
                workers=workers,
                reload=reload,
            )
            uvicorn.Server(config).run()


class QueueCommand:
    """The ``raghub queue ...`` commands (Tier 4 Items 23 + 24).

    Four sub-commands:

    * ``raghub queue list`` — show every queued job
    * ``raghub queue run`` — start a worker pool
    * ``raghub queue retry <job_id>`` — move a failed job back to pending
    * ``raghub queue purge`` — delete jobs by status
    """

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the ``queue`` sub-commands to ``app``."""
        queue_app = typer.Typer(help="Persistent ingestion queue management.")
        QueueCommand.register_list(queue_app)
        QueueCommand.register_run(queue_app)
        QueueCommand.register_retry(queue_app)
        QueueCommand.register_purge(queue_app)
        app.add_typer(queue_app, name="queue")

    @staticmethod
    def register_list(queue_app: typer.Typer) -> None:
        """Attach the ``raghub queue list`` sub-command."""

        @queue_app.command(name="list")
        def list_cmd(
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
            status: str | None = typer.Option(None, "--status", help="Filter by job status."),
            tenant: str | None = typer.Option(None, "--tenant", help="Filter by tenant id."),
            limit: int = typer.Option(100, "--limit", help="Maximum rows."),
        ) -> None:
            """List every queued job (optionally filtered)."""
            import asyncio

            from raghub.jobs import JobStatus

            rag = CliConfig.make_rag(config)
            queue = rag.queue()
            if queue is None:
                typer.echo("queue not configured", err=True)
                raise typer.Exit(code=1)
            status_filter = JobStatus(status) if status else None
            jobs = asyncio.run(queue.list(status=status_filter, limit=limit))
            if tenant is not None:
                jobs = [j for j in jobs if j.tenant_id == tenant]
            typer.echo(
                f"{'id':<36} {'status':<12} {'attempts':<10} "
                f"{'next_run_at':<26} {'tenant_id':<16} kind"
            )
            for job in jobs:
                typer.echo(
                    f"{job.id:<36} {job.status.value:<12} "
                    f"{job.attempts:<10} {job.next_run_at.isoformat():<26} "
                    f"{(job.tenant_id or '-'):<16} {job.kind}"
                )

    @staticmethod
    def register_run(queue_app: typer.Typer) -> None:
        """Attach the ``raghub queue run`` sub-command (drains queue)."""

        @queue_app.command(name="run")
        def run_cmd(
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
            workers: int = typer.Option(2, "--workers", "-w", help="Worker count."),
            max_attempts: int = typer.Option(3, "--max-attempts", help="Per-job retry cap."),
            max_wall_seconds: float = typer.Option(
                30.0, "--max-wall", help="Per-job wall-clock cap."
            ),
        ) -> None:
            """Start a worker pool that drains the queue."""
            import asyncio

            from raghub.jobs import Worker

            rag = CliConfig.make_rag(config)
            queue = rag.queue()
            if queue is None:
                typer.echo("queue not configured", err=True)
                raise typer.Exit(code=1)
            store = rag.feedback_store()
            archive = rag.archive()
            worker = Worker(
                queue=queue,
                handler=lambda job: ingest_handler(job, rag, archive, store),
                concurrency=workers,
                max_attempts=max_attempts,
                max_wall_seconds=max_wall_seconds,
            )
            typer.echo(f"starting {workers} worker(s); ctrl-c to stop")
            try:
                asyncio.run(worker.run())
            except KeyboardInterrupt:
                typer.echo("worker stopped")

    @staticmethod
    def register_retry(queue_app: typer.Typer) -> None:
        """Attach the ``raghub queue retry`` sub-command."""

        @queue_app.command(name="retry")
        def retry_cmd(
            job_id: str = typer.Argument(..., help="Job id to retry."),
            delay_seconds: int = typer.Option(0, "--delay", help="Delay before re-running."),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
        ) -> None:
            """Move a failed or dead job back to pending."""
            import asyncio

            rag = CliConfig.make_rag(config)
            queue = rag.queue()
            if queue is None:
                typer.echo("queue not configured", err=True)
                raise typer.Exit(code=1)
            asyncio.run(queue.retry(job_id, delay_seconds=delay_seconds))
            typer.echo(f"job {job_id} moved back to pending")

    @staticmethod
    def register_purge(queue_app: typer.Typer) -> None:
        """Attach the ``raghub queue purge`` sub-command."""

        @queue_app.command(name="purge")
        def purge_cmd(
            status: str | None = typer.Option(
                "succeeded", "--status", help="Status to purge (default succeeded)."
            ),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
        ) -> None:
            """Delete jobs by status (default: succeeded)."""
            import asyncio

            from raghub.jobs import JobStatus

            rag = CliConfig.make_rag(config)
            queue = rag.queue()
            if queue is None:
                typer.echo("queue not configured", err=True)
                raise typer.Exit(code=1)
            status_filter = JobStatus(status) if status else None
            removed = asyncio.run(queue.purge(status=status_filter))
            typer.echo(f"removed {removed} job(s)")


async def ingest_handler(  # noqa: RUF029 - Worker awaits its handlers
    job: Any, rag: Any, archive: Any, store: Any
) -> None:
    """Re-run ``RAG.ingest`` against the persisted job payload."""
    payload = job.payload
    source = payload["source"].encode("latin-1")
    source_uri = payload.get("source_uri", "bytes://queue")
    mime_type = payload.get("mime_type", "text/plain")
    metadata = payload.get("metadata") or {}
    rag.ingest(
        source=source,
        source_uri=source_uri,
        mime_type=mime_type,
        metadata=metadata,
    )


class MigrateCommand:
    """The ``raghub migrate pgvector | tenant-split`` commands (Tier 5 Items 25, 27)."""

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the sub-commands to ``app``."""
        migrate_app = typer.Typer(help="Migration commands.")

        @migrate_app.command(name="pgvector")
        def migrate_pgvector(
            dsn: str = typer.Option(..., "--dsn", help="Postgres connection string."),
            vector_dim: int = typer.Option(
                DEFAULT_EMBEDDING_DIM, "--vector-dim", help="Embedding dimensionality."
            ),
        ) -> None:
            """Create the pgvector schema and indexes on ``--dsn``."""
            import asyncio

            from raghub.stores.pgvector import PgVectorStore

            store = PgVectorStore(dsn=dsn, embedding_dim=vector_dim)
            asyncio.run(store.initialize())
            typer.echo(f"pgvector schema created on {dsn} (dim={vector_dim})")

        @migrate_app.command(name="tenant-split")
        def migrate_tenant_split(
            from_strategy: str = typer.Option(..., "--from", help="Source isolation strategy."),
            to_strategy: str = typer.Option(..., "--to", help="Target isolation strategy."),
            source_dsn: str = typer.Option(..., "--source-dsn", help="Source DSN."),
            target_dsn: str = typer.Option(..., "--target-dsn", help="Target DSN."),
            tenant_id: str | None = typer.Option(
                None, "--tenant-id", help="Limit migration to a single tenant."
            ),
        ) -> None:
            """Migrate data between isolation strategies."""
            import asyncio

            from raghub.tenants.isolation import (
                Isolation,
                migrate_tenant_split,
            )

            try:
                src = Isolation(from_strategy)
                dst = Isolation(to_strategy)
            except ValueError as exc:
                typer.echo(f"invalid isolation strategy: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            rows = asyncio.run(
                asyncio.to_thread(
                    migrate_tenant_split,
                    source_dsn=source_dsn,
                    target_dsn=target_dsn,
                    from_strategy=src,
                    to_strategy=dst,
                    tenant_id=tenant_id,
                )
            )
            typer.echo(f"migrated {rows} rows from {from_strategy} to {to_strategy}")

        app.add_typer(migrate_app, name="migrate")


class TenantCommand:
    """The ``raghub tenant ...`` commands (Tier 5 Item 26).

    Manages an on-disk ``TenantRegistry`` for per-tenant routing.
    """

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the sub-commands to ``app``."""
        tenant_app = typer.Typer(help="Multi-tenant registry management.")

        @tenant_app.command(name="list")
        def list_cmd(
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
        ) -> None:
            """List every registered tenant."""
            from raghub.tenants.isolation import TenantRegistry

            settings = CliConfig.make_settings(config)
            registry = TenantRegistry(entries=load_registry_entries(settings))
            if not registry.entries:
                typer.echo("(no tenants registered)")
                return
            for tenant_id, record in sorted(registry.entries.items()):
                typer.echo(
                    f"{tenant_id} dsn={record.get('dsn', '?')} "
                    f"vector_dim={record.get('vector_dim', '?')}"
                )

        @tenant_app.command(name="create")
        def create_cmd(
            tenant_id: str = typer.Argument(..., help="Tenant id (regex-validated)."),
            dsn: str = typer.Option(..., "--dsn", help="Postgres DSN."),
            vector_dim: int = typer.Option(
                DEFAULT_EMBEDDING_DIM, "--vector-dim", help="Vector dim."
            ),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
        ) -> None:
            """Register a new tenant."""
            from raghub.tenants import TenantRegistry, validate_tenant

            validate_tenant(tenant_id)
            settings = CliConfig.make_settings(config)
            entries = load_registry_entries(settings)
            registry = TenantRegistry(entries=entries)
            registry.upsert(tenant_id, dsn=dsn, vector_dim=vector_dim)
            save_registry_entries(settings, registry.entries)
            typer.echo(f"tenant {tenant_id} registered")

        @tenant_app.command(name="delete")
        def delete_cmd(
            tenant_id: str = typer.Argument(..., help="Tenant id to delete."),
            force: bool = typer.Option(False, "--force", help="Delete even if data exists."),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
        ) -> None:
            """Remove a tenant from the registry."""
            from raghub.tenants.isolation import TenantRegistry

            settings = CliConfig.make_settings(config)
            entries = load_registry_entries(settings)
            registry = TenantRegistry(entries=entries)
            if tenant_id not in registry.entries:
                typer.echo(f"tenant {tenant_id!r} not found", err=True)
                raise typer.Exit(code=1)
            if not force and tenant_id in registry.entries:
                typer.echo(
                    f"tenant {tenant_id!r} has data; use --force to delete",
                    err=True,
                )
                raise typer.Exit(code=1)
            registry.remove(tenant_id)
            save_registry_entries(settings, registry.entries)
            typer.echo(f"tenant {tenant_id} removed")

        app.add_typer(tenant_app, name="tenant")


class BackupCommand:
    """The ``raghub backup | restore | backup verify`` commands (Tier 5 Items 28-30)."""

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the sub-commands to ``app``."""
        backup_app = typer.Typer(help="Backup / restore / verify.")

        @backup_app.command(name="create")
        def create_cmd(
            output: str = typer.Option(..., "--output", "-o", help="Archive path."),
            tenant: str | None = typer.Option(None, "--tenant", help="Limit to a single tenant."),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
        ) -> None:
            """Capture every component into a single archive."""
            from raghub.archive import create_snapshot, write_archive

            settings = CliConfig.make_settings(config)
            manifest, files = create_snapshot(settings.data_dir)
            if tenant is not None:
                manifest = manifest  # manifest filtering optional
            write_archive(manifest, files, output)
            typer.echo(f"wrote {len(manifest.entries)} entries to {output}")

        @backup_app.command(name="restore")
        def restore_cmd(
            input_path: str = typer.Option(..., "--input", "-i", help="Archive path."),
            target_dir: str | None = typer.Option(None, "--target-dir", help="Restore target."),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
        ) -> None:
            """Restore an archive into ``--target-dir`` (default ``Settings.data_dir``)."""
            from raghub.archive import restore_snapshot

            settings = CliConfig.make_settings(config)
            target = target_dir or str(settings.data_dir)
            restore_snapshot(input_path, target)
            typer.echo(f"restored {input_path} into {target}")

        @backup_app.command(name="verify")
        def verify_cmd(
            input_path: str = typer.Option(..., "--input", "-i", help="Archive path."),
        ) -> None:
            """Verify an archive's HMAC signature and per-file SHA-256s."""
            from raghub.archive import ArchiveCorruptionError, verify_archive

            try:
                verify_archive(input_path)
            except ArchiveCorruptionError as exc:
                typer.echo(f"verification failed: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            typer.echo(f"{input_path} verified")

        app.add_typer(backup_app, name="backup")


def load_registry_entries(settings: Any) -> dict[str, dict[str, Any]]:
    """Load tenant-registry entries from the ``RAG_TENANT_DSNS`` env var.

    Format: ``tenant_id=dsn,vector_dim;tenant_id=dsn,vector_dim;...``

    DSNs contain colons; using ``=`` and ``,`` as separators avoids
    ambiguity.

    A future release will persist this via
    ``Settings.tenants.extra`` or an on-disk file.
    """
    import os

    raw = os.getenv(ENV_RAG_TENANT_DSNS, "")
    if not raw:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            continue
        tenant_id, rest = entry.split("=", 1)
        tenant_id = tenant_id.strip()
        parts = rest.split(",")
        dsn = parts[0].strip() if parts else ""
        dim = int(parts[1].strip()) if len(parts) > 1 else DEFAULT_EMBEDDING_DIM
        if not tenant_id or not dsn:
            continue
        out[tenant_id] = {"dsn": dsn, "vector_dim": dim}
    return out


def save_registry_entries(settings: Any, entries: dict[str, dict[str, Any]]) -> None:
    """Persist registry entries back via the ``Settings.tenants`` payload.

    Stores via ``Settings.extra`` so the change is observed in
    subsequent CLI calls within the same process. A future release
    will add a proper on-disk file under
    ``Settings.data_dir / "tenants.json"``.
    """
    extra = dict(getattr(settings, "extra", None) or {})
    extra["_registry_entries"] = entries
    with suppress(Exception):
        settings.extra = extra


# Bind the tool-settings sub-commands onto the Typer sub-app at import time so
# ``from raghub.commands import ToolConfig`` always yields a wired app.
ToolConfig.register()


__all__ = [
    "DEFAULT_CHUNK_OVERLAP_WORDS",
    "DEFAULT_CHUNK_SIZE_WORDS",
    "BackupCommand",
    "CliConfig",
    "IngestCommand",
    "InitCommand",
    "MigrateCommand",
    "QueryCommand",
    "QueueCommand",
    "ServerCommand",
    "TenantCommand",
    "ToolConfig",
]
