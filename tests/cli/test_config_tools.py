"""Phase 8.4 — ``raghub config tools ...`` CLI subcommands."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

import pytest

from raghub.auth.user_store import SqliteUserStore


def test_config_tools_parser_parses_all_subcommands() -> None:
    from raghub.cli.main import build_parser

    parser = build_parser()
    list_ns = parser.parse_args(["config", "tools", "list", "--email", "a@b.c"])
    assert list_ns.tools_command == "list"

    set_ns = parser.parse_args(
        [
            "config",
            "tools",
            "set",
            "--email",
            "a@b.c",
            "--json",
            '{"agent_enabled": true}',
        ]
    )
    assert set_ns.tools_command == "set"
    assert set_ns.payload == '{"agent_enabled": true}'

    unset_ns = parser.parse_args(["config", "tools", "unset", "--email", "a@b.c"])
    assert unset_ns.tools_command == "unset"


def test_config_tools_set_filters_unknown_keys() -> None:
    """Unknown keys are dropped silently so a typo can't break startup."""
    from raghub.cli import config_cmd

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteUserStore(Path(tmp) / "users.db")
            await store.initialize()
            await store.create_user("a@b.c", "password")
            user_id = (await store.get_by_email("a@b.c")).user_id

            # Patch async_load_store to return our fresh store.
            async def make_store():
                return store

            config_cmd.async_load_store = make_store  # type: ignore[attr-defined]
            ns = argparse.Namespace(
                email="a@b.c",
                payload='{"agent_enabled": true, "BOGUS_KEY": "ignore"}',
            )
            assert config_cmd.handle_set(ns) == 0
            prefs = await store.get_pref(user_id, "tool_settings")
            assert "agent_enabled" in prefs
            assert "BOGUS_KEY" not in prefs

    asyncio.run(runner())


def test_config_tools_set_rejects_non_json() -> None:
    from raghub.cli import config_cmd

    ns = argparse.Namespace(email="a@b.c", payload="not json {")
    assert config_cmd.handle_set(ns) == 2


def test_config_tools_set_rejects_non_object() -> None:
    from raghub.cli import config_cmd

    ns = argparse.Namespace(email="a@b.c", payload="[1, 2, 3]")
    assert config_cmd.handle_set(ns) == 2


def test_config_tools_unset_removes_blob() -> None:
    """``unset`` deletes the stored tool_settings."""
    from raghub.cli import config_cmd

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteUserStore(Path(tmp) / "users.db")
            await store.initialize()
            await store.create_user("a@b.c", "password")
            user_id = (await store.get_by_email("a@b.c")).user_id
            await store.set_pref(user_id, "tool_settings", {"agent_enabled": True})

            async def make_store():
                return store

            config_cmd.async_load_store = make_store  # type: ignore[attr-defined]
            assert config_cmd.handle_unset(argparse.Namespace(email="a@b.c")) == 0
            assert await store.get_pref(user_id, "tool_settings") is None

    asyncio.run(runner())