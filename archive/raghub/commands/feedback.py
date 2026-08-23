"""``raghub feedback ...`` sub-commands.

Extracted from raghub.commands.__init__ to keep the main commands
module under the AGENTS.md §743-754 size guideline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer

from raghub.commands.cli_config import CliConfig


class FeedbackCommand:
    """The ``raghub feedback ...`` commands (Tier 3 Item 20).

    Two sub-commands:

    * ``raghub feedback export --jsonl <path> [--tenant <id>]``
    * ``raghub feedback stats [--tenant <id>]``
    """

    @staticmethod
    def register(app: typer.Typer) -> None:
        """Attach the ``feedback`` sub-commands to ``app``."""
        feedback_app = typer.Typer(help="Feedback capture and export.")

        @feedback_app.command(name="export")
        def export_cmd(
            jsonl_path: str = typer.Option(..., "--jsonl", help="Output JSONL file path."),
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
            tenant: str | None = typer.Option(None, "--tenant", help="Filter by tenant id."),
        ) -> None:
            """Export every feedback record as JSONL to ``--jsonl``."""
            rag = CliConfig.make_rag(config)
            store = rag.feedback_store()
            if store is None:
                typer.echo("feedback_store not configured", err=True)
                raise typer.Exit(code=1)
            records = FeedbackCommand.load_records(store, tenant)

            written = 0
            with open(jsonl_path, "w", encoding="utf-8") as handle:
                for record in records:
                    if record.tenant_id != tenant:
                        continue
                    payload = FeedbackCommand.serialise_record(record)
                    handle.write(json.dumps(payload, default=str) + "\n")
                    written += 1
            typer.echo(f"wrote {written} feedback records to {jsonl_path}")

        @feedback_app.command(name="stats")
        def stats_cmd(
            config: str | None = typer.Option(
                None, "--config", "-c", help="Optional YAML/TOML config path."
            ),
            tenant: str | None = typer.Option(None, "--tenant", help="Filter by tenant id."),
        ) -> None:
            """Print aggregate feedback counts."""
            rag = CliConfig.make_rag(config)
            store = rag.feedback_store()
            if store is None:
                typer.echo("feedback_store not configured", err=True)
                raise typer.Exit(code=1)
            aggregate = asyncio.run(store.aggregate(tenant))
            typer.echo(
                f"tenant={aggregate.tenant_id} positive={aggregate.positive} "
                f"negative={aggregate.negative} neutral={aggregate.neutral}"
            )

        app.add_typer(feedback_app, name="feedback")

    @staticmethod
    def load_records(store: Any, tenant: str | None) -> list[Any]:
        """Load all feedback records from ``store`` for the optional tenant filter.

        Empty-tenant-string is used as a wildcard by the SQL
        implementation; here we explicitly request the unfiltered
        list when ``tenant`` is None.
        """
        if tenant is not None:
            return asyncio.run(store.list_for_tenant(tenant))
        return asyncio.run(store.list_for_tenant(""))

    @staticmethod
    def serialise_record(record: Any) -> dict[str, Any]:
        """Project a feedback record into a JSON-serialisable dict."""
        return {
            "id": record.id,
            "session_id": record.session_id,
            "query_id": record.query_id,
            "chunk_id": record.chunk_id,
            "answer_id": record.answer_id,
            "user_id": record.user_id,
            "tenant_id": record.tenant_id,
            "rating": int(record.rating),
            "comment": record.comment,
            "created_at": record.created_at.isoformat(),
            "metadata": record.metadata,
        }


__all__ = ["FeedbackCommand"]
