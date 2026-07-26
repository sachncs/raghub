"""Top-level ``raghub`` CLI — Typer app composition + entry point.

Single Typer app that pulls in every command via the ``register(app)``
pattern. Each command lives in its own module under
:mod:`raghub.cli.<command>` so the structure mirrors the public
surface one-to-one.

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
import importlib.util

import typer

app = typer.Typer(
    name="raghub",
    help="RAGHub — production-grade multi-user retrieval-augmented generation.",
    no_args_is_help=True,
)

from raghub.cli.format import make_rag, write_json

# Sub-trees — these are the only Typer-group attachments. Each gets its own
# help subtree under the main app.
from raghub.cli.config import app as config_app
from raghub.evaluation.cli import app as eval_app
app.add_typer(config_app, name="config")
app.add_typer(eval_app, name="eval")

# Flat commands — registered as `@app.command(name="...")` directly on `app`.
from raghub.cli.ingest import register as register_ingest
from raghub.cli.init import register as register_init
from raghub.cli.query import register as register_query
from raghub.cli.server import register as register_run
register_query(app)
register_ingest(app)
register_init(app)
register_run(app)


@app.command(name="health")
def health() -> None:
    """Print the framework liveness status as JSON."""
    write_json(make_rag(None).health())


@app.command(name="version")
def version() -> None:
    """Print the installed ``raghub`` package version."""
    try:
        typer.echo(importlib.metadata.version("raghub"))
    except importlib.metadata.PackageNotFoundError:
        typer.echo("unknown")


def main() -> None:
    """Entry point for the ``raghub`` console script."""
    raise typer.Exit(app())


__all__ = ["app", "health", "main", "version"]