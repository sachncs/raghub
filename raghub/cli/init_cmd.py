"""``raghub init`` — emit a starter YAML config."""

from __future__ import annotations

import argparse

from loguru import logger as loguru_logger

SAMPLE_CONFIG = """# RAGHub configuration — adjust to your environment.
environment: development
data_dir: ./data
chunk_size_words: 800
chunk_overlap_words: 100
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


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``init`` subcommand."""
    parser = subparsers.add_parser("init", help="print a sample config")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="write the sample to this path instead of stdout",
    )
    parser.set_defaults(handler=run_subcommand)


def run_subcommand(args: argparse.Namespace) -> int:
    """Write or print the sample config.

    Args:
        args: Parsed CLI args with optional ``--output`` path.

    Returns:
        ``0`` on success.
    """
    from pathlib import Path

    if args.output:
        Path(args.output).write_text(SAMPLE_CONFIG, encoding="utf-8")
        loguru_logger.info("cli.init.wrote", path=str(args.output))
    else:
        loguru_logger.info(SAMPLE_CONFIG)
    return 0
