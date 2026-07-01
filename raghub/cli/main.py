"""Top-level CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from raghub.cli import eval_cmd, init_cmd, ingest_cmd, query_cmd, system
from raghub.cli.rate_limiter import CLIRateLimiter, RateLimitExceeded

_limiter = CLIRateLimiter()

# Commands that are exempt from rate limiting (e.g. health, version).
_RATE_LIMIT_EXEMPT = frozenset({"health", "version"})


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

    command = args.command
    if command not in _RATE_LIMIT_EXEMPT:
        try:
            _limiter.check(command)
        except RateLimitExceeded as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return int(handler(args))


if __name__ == "__main__":
    main()
