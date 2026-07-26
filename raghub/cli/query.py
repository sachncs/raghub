"""``raghub query "..."`` — ask a question against the index.

Imports nothing Typer-specific; :func:`register` attaches this command
to the main Typer app declared in :mod:`raghub.cli.main`.
"""

from __future__ import annotations

import typer


def register(app: "typer.Typer") -> None:
    """Attach the ``query`` command to ``app``."""

    @app.command(name="query")
    def query(
        question: str = typer.Argument(..., help="The question to ask."),
        config: str | None = typer.Option(None, "--config", "-c", help="Optional YAML/TOML config path."),
        top_k: int = typer.Option(5, "--top-k", "-k", help="Number of hits to retrieve."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Ask a question and print the answer + citations."""
        from raghub.cli.format import make_rag, write_json

        rag = make_rag(config)
        response = rag.query(question, top_k=top_k)
        if json_output:
            write_json(
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
                typer.echo(f"  - {c.document_id}#{c.chunk_id} (score={c.score:.3f})")


__all__ = ["register"]
