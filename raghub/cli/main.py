"""Top-level CLI entrypoint."""

from __future__ import annotations

import argparse

from raghub.cli import eval_cmd, init_cmd, ingest_cmd, query_cmd, system


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(prog="raghub", description="RAGHub CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_cmd.add_parser(subparsers)
    ingest_cmd.add_parser(subparsers)
    query_cmd.add_parser(subparsers)
    eval_cmd.add_parser(subparsers)
    system.add_parser(subparsers)
    return parser


def main() -> int:
    """Entry point for ``python -m raghub.cli``."""
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    main()
