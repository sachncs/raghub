"""``raghub query "..."`` command."""

from __future__ import annotations

import argparse

from raghub.api.rag import RAG
from raghub.cli.common import write_json


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``query`` subcommand."""
    parser = subparsers.add_parser("query", help="ask a question")
    parser.add_argument("question", help="question to ask")
    parser.add_argument(
        "--config",
        default=None,
        help="optional path to a YAML/TOML config file",
    )
    parser.add_argument("--top-k", type=int, default=5, help="top-k hits to retrieve")
    parser.set_defaults(handler=run_subcommand)


def run_subcommand(args: argparse.Namespace) -> int:
    """Run a synchronous query against the RAG facade.

    Args:
        args: Parsed CLI args (``question``, ``--top-k``, ``--config``).

    Returns:
        ``0`` on success.
    """
    rag = RAG.from_config(args.config) if args.config else RAG()
    response = rag.query(args.question, top_k=args.top_k)
    write_json(
        {
            "answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "source_chunks": [s.model_dump() for s in response.source_chunks],
            "metadata": response.metadata,
        }
    )
    return 0
