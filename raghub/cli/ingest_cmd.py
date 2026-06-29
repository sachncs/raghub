"""``raghub ingest PATH`` command."""

from __future__ import annotations

import argparse

from raghub.cli._common import print_json
from raghub.api.rag import RAG


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``ingest`` subcommand."""
    parser = subparsers.add_parser("ingest", help="ingest a file or directory")
    parser.add_argument("path", help="file or directory to ingest")
    parser.add_argument(
        "--config",
        default=None,
        help="optional path to a YAML/TOML config file",
    )
    parser.add_argument(
        "--async",
        dest="async_job",
        action="store_true",
        help="submit to the background service and return a job id",
    )
    parser.set_defaults(handler=run_subcommand)


def run_subcommand(args: argparse.Namespace) -> int:
    """Run an ingest job against the RAG facade.

    Args:
        args: Parsed CLI args (``path``, ``--config``, ``--async``).

    Returns:
        ``0`` on success.
    """
    if args.config:
        rag = RAG.from_config(args.config)
    else:
        rag = RAG()

    if args.async_job:
        job_id = rag.ingest_async(args.path)
        print_json({"job_id": job_id})
        return 0

    result = rag.ingest(args.path)
    if result.outputs.get("batch"):
        # Directory ingest: print per-file results.
        batch_payload: list = [r.model_dump(mode="json") for r in result.outputs["batch"]]
        print_json(batch_payload)
    else:
        print_json(result.model_dump(mode="json"))
    return 0
