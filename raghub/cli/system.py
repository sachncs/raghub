"""``raghub health`` and ``raghub version`` commands."""

from __future__ import annotations

import argparse

from loguru import logger as loguru_logger

from raghub.api.rag import RAG
from raghub.cli.common import write_json


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``health`` and ``version`` subcommands."""
    health = subparsers.add_parser("health", help="liveness probe")
    health.set_defaults(handler=handle_health)

    version = subparsers.add_parser("version", help="print the package version")
    version.set_defaults(handler=handle_version)


def handle_health(_: argparse.Namespace) -> int:
    """Run a liveness probe and print the health status as JSON.

    Args:
        _: Unused argparse namespace (required by the handler protocol).

    Returns:
        ``0`` on success.
    """
    rag = RAG()
    write_json(rag.health())
    return 0


def handle_version(_: argparse.Namespace) -> int:
    """Print the package version via the loguru logger.

    Reads via :func:`importlib.metadata.version`. When the package
    is not installed in editable mode the metadata is unavailable;
    in that case we print ``"unknown"`` and exit 0 so the command
    is safe to run in any environment.

    Args:
        _: Unused argparse namespace (required by the handler protocol).

    Returns:
        ``0`` on success.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    try:
        loguru_logger.info("cli.version", version=package_version("raghub"))
    except PackageNotFoundError:
        loguru_logger.info("cli.version", version="unknown")
    return 0
