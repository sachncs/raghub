"""CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json

from raghub.core.container import build_application


def main() -> None:
    """Run CLI commands."""

    parser = argparse.ArgumentParser(prog="raghub")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login")
    login_parser.add_argument("email")
    login_parser.add_argument("password", nargs="?", default="")

    subparsers.add_parser("health")

    args = parser.parse_args()
    app = asyncio.run(build_application())

    if args.command == "login":
        print(json.dumps(asyncio.run(app.login(args.email, args.password)).model_dump(mode="json"), indent=2))
    elif args.command == "health":
        print(json.dumps(app.health(), indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
