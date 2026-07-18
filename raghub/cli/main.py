"""Top-level CLI entrypoint."""

from __future__ import annotations

import argparse

from loguru import logger as loguru_logger

from raghub.cli import eval_cmd, ingest_cmd, init_cmd, query_cmd, system
from raghub.cli.rate_limiter import CLIRateLimiter, RateLimitExceeded

CLI_LIMITER = CLIRateLimiter()

# Commands that are exempt from rate limiting (e.g. health, version).
RATE_LIMIT_EXEMPT_COMMANDS = frozenset({"health", "version"})


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

    command = getattr(args, "command", None)
    if command is not None and command not in RATE_LIMIT_EXEMPT_COMMANDS:
        try:
            CLI_LIMITER.check(command)
        except RateLimitExceeded as exc:
            loguru_logger.error("cli.rate_limit_exceeded", command=command, error=str(exc))
            return 1

    return int(handler(args))


if __name__ == "__main__":
    main()
