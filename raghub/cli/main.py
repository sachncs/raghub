"""Top-level ``raghub`` CLI — Typer app composition + entry point.

Single Typer app that pulls in every command from :mod:`raghub.cli.helper`.
Each command is a class with a ``register(app)`` method; this module
calls them in order.

Usage (after install)::

    raghub --help
    raghub query "what is revenue?"
    raghub ingest ./docs/report.pdf
    raghub init -o raghub.yaml
    raghub run --port 8000
    raghub health
    raghub config tools list --email alice@acme.com
    raghub eval financebench --examples 5
"""

from __future__ import annotations

import importlib.metadata

import typer

from raghub.cli.helper import (
    CliConfig,
    InitCommand,
    IngestCommand,
    QueryCommand,
    ServerCommand,
    ToolConfig,
)
from raghub.evaluation.cli import app as eval_app
from raghub.utils import capture

app = typer.Typer(
    name="raghub",
    help="RAGHub — production-grade multi-user retrieval-augmented generation.",
    no_args_is_help=True,
)

# Sub-trees — these are the only Typer-group attachments. Each gets its own
# help subtree under the main app.
ToolConfig.register()
app.add_typer(ToolConfig.app, name="config")
app.add_typer(eval_app, name="eval")

# Flat commands — registered as `@app.command(name="...")` directly on `app`.
QueryCommand.register(app)
IngestCommand.register(app)
InitCommand.register(app)
ServerCommand.register(app)


@app.command(name="health")
def health() -> None:
    """Print the framework liveness status as JSON."""
    CliConfig.write_json(CliConfig.make_rag(None).health())


@app.command(name="version")
def version() -> None:
    """Print the installed ``raghub`` package version."""
    version_str, error = capture(
        importlib.metadata.version, "raghub"
    )
    typer.echo(version_str if error is None else "unknown")


def main() -> None:
    """Entry point for the ``raghub`` console script."""
    raise typer.Exit(app())
