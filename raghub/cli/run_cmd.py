"""``raghub run`` — start the FastAPI server as a foreground daemon.

This subcommand exists primarily for operators who want a single
binary entrypoint: ``raghub run`` boots the application container and
serves it via uvicorn without an extra ``uvicorn`` invocation. For
production, use the Docker image (see ``docker-compose.yml``) or
launch uvicorn directly.
"""

from __future__ import annotations

import argparse

from loguru import logger as loguru_logger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run`` subcommand.

    Args:
        subparsers: The argparse subparser registry.
    """
    parser = subparsers.add_parser("run", help="run the API server (foreground)")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default 8000)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="uvicorn worker count (default 1; production should set higher)",
    )
    parser.set_defaults(handler=run_subcommand)


def run_subcommand(args: argparse.Namespace) -> int:
    """Run uvicorn in the foreground.

    Args:
        args: Parsed CLI args (``--host``, ``--port``, ``--workers``).

    Returns:
        Uvicorn's process exit code (typically ``0`` on a clean
        ``SIGINT`` shutdown and ``1`` on a bind error).
    """
    import uvicorn

    loguru_logger.info(
        "cli.run.starting",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
    config = uvicorn.Config(
        "raghub.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
    server = uvicorn.Server(config)
    server.run()
    return int(server.should_exit and 0 or 1)


__all__ = ["add_parser", "run_subcommand"]