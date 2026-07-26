"""``raghub config tools`` subcommands (Phase 8.4).

Read / write the ``tool_settings`` preference blob via the user
store. Operates on the same backing database the API uses; the
preferences surface and the CLI are interchangeable.
"""

from __future__ import annotations

import argparse
import json

from loguru import logger as loguru_logger

from raghub.auth.user_store import SqliteUserStore
from raghub.cli.common import load_settings_or_path, write_json

_TOOL_KEYS = (
    "tools_enabled",
    "agent_enabled",
    "web",
    "graph",
    "summaries",
    "reranker",
    "long_context_pass",
    "query_transforms",
    "max_steps",
)


def resolve_user_id(store: SqliteUserStore, email: str) -> str:
    """Look up the user id by email or raise ``SystemExit(2)``.

    Args:
        store: The :class:`SqliteUserStore` to query.
        email: The user's email address.

    Returns:
        The user id.

    Raises:
        SystemExit: When no user with that email exists.
    """
    record = run_async(store.get_by_email(email))
    if record is None:
        raise SystemExit(f"no user with email {email!r}")
    return record.user_id


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``raghub config tools ...`` subcommands."""
    config = subparsers.add_parser("config", help="manage per-user tool defaults")
    config_sub = config.add_subparsers(dest="config_command", required=True)

    # ``raghub config tools <list|set|unset>``
    tools = config_sub.add_parser("tools", help="manage the tool_settings blob")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)

    list_cmd = tools_sub.add_parser(
        "list", help="print the tool_settings blob for a user"
    )
    list_cmd.add_argument("--email", required=True, help="user email")
    list_cmd.set_defaults(handler=handle_list)

    set_cmd = tools_sub.add_parser(
        "set", help="merge a JSON object into tool_settings"
    )
    set_cmd.add_argument("--email", required=True, help="user email")
    set_cmd.add_argument(
        "--json",
        dest="payload",
        required=True,
        help='JSON object, e.g. \'{"agent_enabled": true, "reranker": "bge"}\'',
    )
    set_cmd.set_defaults(handler=handle_set)

    unset_cmd = tools_sub.add_parser(
        "unset", help="delete the tool_settings blob entirely"
    )
    unset_cmd.add_argument("--email", required=True, help="user email")
    unset_cmd.set_defaults(handler=handle_unset)


async def async_load_store() -> SqliteUserStore:
    """Build and initialise the user store for the active profile.

    Returns:
        The initialised :class:`SqliteUserStore`.
    """
    from pathlib import Path

    settings = load_settings_or_path(None)
    db_path = Path(settings.data_dir) / "users.db"
    store = SqliteUserStore(db_path)
    await store.initialize()
    return store


def run_async(coro):
    """Run ``coro`` in a fresh event loop, even when one is already active.

    Splits the difference between ``asyncio.run`` (which forbids
    re-entry) and ``asyncio.get_event_loop().run_until_complete``
    (which is deprecated under 3.12+). Tests in this module use
    ``pytest-asyncio``'s event loop, so handlers need a way to
    schedule a coroutine without re-entering it.

    Args:
        coro: The coroutine to drive.

    Returns:
        The coroutine's return value.

    Raises:
        BaseException: Whatever the coroutine raised, on its
            private thread.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # An event loop is already running (test). Drive the coroutine on
    # a private thread to avoid re-entrancy.
    import threading

    result: list = [None]
    error: list = [None]

    def runner() -> None:
        """Run ``coro`` on the private thread, surfacing errors via ``error``."""
        try:
            result[0] = asyncio.run(coro)
        except BaseException as exc:
            error[0] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error[0] is not None:
        raise error[0]
    return result[0]


def handle_list(args: argparse.Namespace) -> int:
    """Print the current tool_settings blob as JSON."""

    async def runner() -> SqliteUserStore:
        return await async_load_store()

    store = run_async(runner())
    user_id = resolve_user_id(store, args.email)
    prefs = run_async(store.get_prefs(user_id))
    write_json({"email": args.email, "tool_settings": prefs.get("tool_settings", {})})
    return 0


def handle_set(args: argparse.Namespace) -> int:
    """Merge ``args.payload`` (JSON) into the user's ``tool_settings``."""
    try:
        patch = json.loads(args.payload)
    except ValueError as exc:
        loguru_logger.error("cli.config.tools.set.bad_json", error=str(exc))
        return 2
    if not isinstance(patch, dict):
        loguru_logger.error("cli.config.tools.set.not_object", type=type(patch).__name__)
        return 2

    async def runner() -> None:
        store = await async_load_store()
        user_id = resolve_user_id(store, args.email)
        existing = await store.get_pref(user_id, "tool_settings") or {}
        if not isinstance(existing, dict):
            existing = {}
        merged = {**existing, **{k: v for k, v in patch.items() if k in _TOOL_KEYS}}
        await store.set_pref(user_id, "tool_settings", merged)
        write_json({"email": args.email, "tool_settings": merged})

    run_async(runner())
    return 0


def handle_unset(args: argparse.Namespace) -> int:
    """Delete the ``tool_settings`` key for the user."""

    async def runner() -> None:
        store = await async_load_store()
        user_id = resolve_user_id(store, args.email)
        await store.delete_pref(user_id, "tool_settings")
        write_json({"email": args.email, "tool_settings": None})

    run_async(runner())
    return 0


__all__ = ["add_parser", "handle_list", "handle_set", "handle_unset"]