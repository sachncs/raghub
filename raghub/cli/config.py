"""``raghub config tools ...`` — per-user tool defaults on the CLI.

Configuration commands live with the configuration package. The
public surface is the nested Typer app:

* ``raghub config tools list`` — print the current tool_settings blob.
* ``raghub config tools set`` — merge a JSON object into tool_settings.
* ``raghub config tools unset`` — delete tool_settings entirely.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from raghub.auth import SqliteUserStore
from raghub.cli.format import read_settings, write_json

app = typer.Typer(help="Configuration commands.", no_args_is_help=True)
tools = typer.Typer(help="Manage the tool_settings blob.", no_args_is_help=True)
app.add_typer(tools, name="tools")

TOOL_KEYS: tuple[str, ...] = (
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


async def load_store() -> Any:
    """Build and initialise the user store for the active profile.

    Returns:
        The initialised :class:`raghub.auth.user_store.SqliteUserStore`.
    """
    settings = read_settings(None)
    db_path = Path(settings.data_dir) / "users.db"
    store = SqliteUserStore(db_path)
    await store.initialize()
    return store


def run_coro(coro: Any) -> Any:
    """Run an awaitable in a fresh event loop."""
    return asyncio.run(coro)


@tools.command(name="list")
def list_cmd(email: str = typer.Option(..., "--email", help="User email.")) -> None:
    """Print the tool_settings blob for ``--email`` as JSON."""

    async def runner() -> None:
        store = await load_store()
        user = await store.get_by_email(email)
        if user is None:
            raise typer.BadParameter(f"no user with email {email!r}")
        prefs = await store.get_prefs(user.user_id)
        write_json({"email": email, "tool_settings": prefs.get("tool_settings", {})})

    run_coro(runner())


@tools.command(name="set")
def set_cmd(
    email: str = typer.Option(..., "--email", help="User email."),
    payload: str = typer.Option(
        ...,
        "--json",
        help='JSON object, e.g. \'{"agent_enabled": true, "reranker": "bge"}\'.',
    ),
) -> None:
    """Merge the JSON ``--json`` payload into the user's tool_settings (unknown keys dropped)."""
    patch, json_error = capture(json.loads, payload)
    if json_error is not None:
        raise typer.BadParameter(f"invalid JSON: {json_error}") from json_error
    if not isinstance(patch, dict):
        raise typer.BadParameter("--json must be a JSON object")

    async def runner() -> None:
        store = await load_store()
        user = await store.get_by_email(email)
        if user is None:
            raise typer.BadParameter(f"no user with email {email!r}")
        existing = await store.get_pref(user.user_id, "tool_settings") or {}
        if not isinstance(existing, dict):
            existing = {}
        merged = {**existing, **{k: v for k, v in patch.items() if k in TOOL_KEYS}}
        await store.set_pref(user.user_id, "tool_settings", merged)
        write_json({"email": email, "tool_settings": merged})

    run_coro(runner())


@tools.command(name="unset")
def unset_cmd(email: str = typer.Option(..., "--email", help="User email.")) -> None:
    """Delete the tool_settings blob for ``--email``."""

    async def runner() -> None:
        store = await load_store()
        user = await store.get_by_email(email)
        if user is None:
            raise typer.BadParameter(f"no user with email {email!r}")
        await store.delete_pref(user.user_id, "tool_settings")
        write_json({"email": email, "tool_settings": None})

    run_coro(runner())


__all__ = ["TOOL_KEYS", "app", "load_store", "tools"]