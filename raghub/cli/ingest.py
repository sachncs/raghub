"""``raghub ingest PATH`` — convert + chunk + embed + index.

Registers the ``ingest`` command on the main Typer app via :func:`register`.
"""

from __future__ import annotations

import typer

from raghub.cli.format import make_rag, write_json


def register(app: "typer.Typer") -> None:
    """Attach the ``ingest`` command to ``app``."""

    @app.command(name="ingest")
    def ingest(
        path: str = typer.Argument(..., help="Path to a file or directory."),
        config: str | None = typer.Option(None, "--config", "-c", help="Optional YAML/TOML config path."),
        background: bool = typer.Option(False, "--async", help="Submit to the background job service and print a job id."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Ingest ``PATH`` synchronously, or submit it as a background job with ``--async``."""
        rag = make_rag(config)
        if background:
            job_id = rag.ingest_async(path)
            if json_output:
                write_json({"job_id": job_id})
            else:
                typer.echo(f"submitted job {job_id}")
            return
        result = rag.ingest(path)
        batch = result.outputs.get("batch")
        if batch:
            payload = [r.model_dump(mode="json") for r in batch]
        else:
            payload = result.model_dump(mode="json")
        if json_output:
            write_json(payload)
            return
        success = getattr(payload, "get", lambda *_: None)("success")
        if success is not None:
            typer.echo(f"{'OK' if success else 'FAIL'}: {path}")
        elif isinstance(payload, list):
            typer.echo(f"ingested batch of {len(payload)} item(s)")
        else:
            typer.echo(f"ingested {path}")


__all__ = ["register"]
