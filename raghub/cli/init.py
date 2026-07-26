"""``raghub init`` — emit a starter YAML config.

Registers the ``init`` command on the main Typer app via :func:`register`.
"""

from __future__ import annotations

from pathlib import Path

import typer

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


def register(app: "typer.Typer") -> None:
    """Attach the ``init`` command to ``app``."""

    @app.command(name="init")
    def init(
        output: str | None = typer.Option(None, "--output", "-o", help="Write to this path instead of stdout."),
    ) -> None:
        """Print the starter config to stdout, or to ``--output``."""
        if output:
            Path(output).write_text(SAMPLE_CONFIG, encoding="utf-8")
            typer.echo(f"wrote {output}")
        else:
            typer.echo(SAMPLE_CONFIG, nl=False)


