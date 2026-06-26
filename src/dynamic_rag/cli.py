"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json

from dynamic_rag.core.container import build_application


def main() -> None:
    """Run CLI commands."""

    parser = argparse.ArgumentParser(prog="dynamic_rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login")
    login_parser.add_argument("email")

    subparsers.add_parser("health")

    args = parser.parse_args()
    app = build_application()

    if args.command == "login":
        print(json.dumps(app.login(args.email).model_dump(mode="json"), indent=2))
    elif args.command == "health":
        print(json.dumps(app.health(), indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
