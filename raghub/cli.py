"""Console-script entry point for the ``raghub`` CLI.

Assembles the Typer app from the command classes in
:mod:`raghub.cli_commands` and the eval sub-app in
:mod:`raghub.evaluation`. ``main`` is the ``raghub`` console script.
"""

from __future__ import annotations

import importlib.metadata

import typer

from raghub.await_sync import capture
from raghub.cli_commands import (
    CliConfig,
    FeedbackCommand,
    IngestCommand,
    InitCommand,
    QueryCommand,
    QueueCommand,
    ServerCommand,
    ToolConfig,
)
from raghub.evaluation import app as eval_app

__all__ = ["health", "main", "version"]


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
FeedbackCommand.register(app)
QueueCommand.register(app)


@app.command(name="health")
def health() -> None:
    """Print the framework liveness status as JSON."""
    CliConfig.write_json(CliConfig.make_rag(None).health())


@app.command(name="version")
def version() -> None:
    """Print the installed ``raghub`` package version."""
    version_str, error = capture(importlib.metadata.version, "raghub")
    typer.echo(version_str if error is None else "unknown")


def main() -> None:
    """Entry point for the ``raghub`` console script."""
    raise typer.Exit(app())
