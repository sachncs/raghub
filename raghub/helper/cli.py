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
from pathlib import Path
from typing import Any

import typer
import uvicorn
import yaml

from raghub.config import Settings
from raghub.rag import RAG
from raghub.utils import capture
from raghub.utils import write_json as write_json_impl


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
        return Settings(**{k: v for k, v in data.items() if k in Settings.model_fields})

    @staticmethod
    def make_rag(config: str | None) -> Any:
        """Instantiate a :class:`raghub.RAG` from ``config`` or defaults.

        Args:
            config: Optional path to a YAML/TOML config file.

        Returns:
            A configured :class:`raghub.RAG` instance.

        """
        return RAG.from_config(config) if config else RAG()


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
    def register(cls) -> None:
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
            """Merge the JSON ``--json`` payload into the user's tool_settings (unknown keys dropped)."""
            patch, json_error = capture(json.loads, payload)
            if json_error is not None:
                raise typer.BadParameter(f"invalid JSON: {json_error}") from json_error
            if not isinstance(patch, dict):
                raise typer.BadParameter("--json must be a JSON object")

            async def runner() -> None:
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
                    r.model_dump(mode="json") for r in batch
                ]
            else:
                payload = result.model_dump(mode="json")
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
chunk_size_words: 800
chunk_overlap_words: 100
chunker_strategy: recursive
embedding_model_chunker: minishlab/potion-base-8M
embedding_dim: 384
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
                        "citations": [c.model_dump() for c in response.citations],
                        "source_chunks": [s.model_dump() for s in response.source_chunks],
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
            config = uvicorn.Config(
                "raghub.api:AppFactory.create_app",
                factory=True,
                host=host,
                port=port,
                workers=workers,
                reload=reload,
            )
            uvicorn.Server(config).run()


# Bind the tool-settings sub-commands onto the Typer sub-app at import time so
# ``from raghub.helper.cli import ToolConfig`` always yields a wired app.
ToolConfig.register()


__all__ = [
    "CliConfig",
    "IngestCommand",
    "InitCommand",
    "QueryCommand",
    "ServerCommand",
    "ToolConfig",
]
