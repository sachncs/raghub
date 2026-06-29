"""``raghub health`` and ``raghub version`` commands."""

from __future__ import annotations

import argparse

from raghub.api.rag import RAG
from raghub.cli._common import print_json


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``health`` and ``version`` subcommands."""
    health = subparsers.add_parser("health", help="liveness probe")
    health.set_defaults(handler=handle_health)

    version = subparsers.add_parser("version", help="print the package version")
    version.set_defaults(handler=handle_version)


def handle_health(_: argparse.Namespace) -> int:
    rag = RAG()
    print_json(rag.health())
    return 0


def handle_version(_: argparse.Namespace) -> int:
    """Print the package version.

    Reads via :func:`importlib.metadata.version`. When the package
    is not installed in editable mode the metadata is unavailable;
    in that case we print ``"unknown"`` and exit 0 so the command
    is safe to run in any environment.
    """
    from importlib.metadata import PackageNotFoundError, version as _v

    try:
        print(_v("retrieval-augmented-generation"))
    except PackageNotFoundError:
        print("unknown")
    return 0
