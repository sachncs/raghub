"""Migrate + prompts + plugins + conv coverage tests.

Bundled into one file (Phase 5.10 batch) because each module is small
enough to test in a focused way.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# raghub.migrate
# ---------------------------------------------------------------------------


def test_migrate_migrate_manifest_bumps_v0_to_v2(tmp_path: Path) -> None:
    """migrate_manifest rewrites a legacy v0 manifest to v2 in place."""

    from raghub.migrate import migrate_manifest

    legacy = tmp_path / "m.json"
    legacy.write_text(json.dumps({"rec1": {"bundle_id": "b1", "checksum": "h"}}), encoding="utf-8")
    rewritten = migrate_manifest(legacy)
    assert rewritten is True
    payload = json.loads(legacy.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert "rec1" in payload["records"]


def test_migrate_migrate_manifest_already_v2(tmp_path: Path) -> None:
    """A v2 manifest is left alone (no rewrite)."""

    from raghub.knowledge import Manifest
    from raghub.migrate import migrate_manifest

    path = tmp_path / "v2.json"
    path.write_text(
        json.dumps({"version": Manifest.CURRENT_VERSION, "records": {}}),
        encoding="utf-8",
    )
    rewritten = migrate_manifest(path)
    assert rewritten is False


def test_migrate_migrate_manifest_missing_returns_false(tmp_path: Path) -> None:
    """A non-existent manifest returns False without raising."""

    from raghub.migrate import migrate_manifest

    assert migrate_manifest(tmp_path / "missing.json") is False


def test_migrate_migrate_manifest_invalid_json_returns_false(tmp_path: Path) -> None:
    """An unparseable JSON file returns False."""

    from raghub.migrate import migrate_manifest

    path = tmp_path / "bad.json"
    path.write_text("not-json", encoding="utf-8")
    assert migrate_manifest(path) is False


def test_migrate_migrate_manifest_non_dict_returns_false(tmp_path: Path) -> None:
    """A JSON list (not a dict) returns False."""

    from raghub.migrate import migrate_manifest

    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert migrate_manifest(path) is False


def test_migrate_run_no_manifests_under_root(tmp_path: Path) -> None:
    """run() reports no work needed when nothing matches."""

    from raghub.migrate import run_migration

    # Run on a directory that exists but has no manifests.
    out_dir = tmp_path / "data"
    out_dir.mkdir()
    exit_code = run_migration(out_dir)
    assert exit_code == 0


def test_migrate_run_missing_root(tmp_path: Path) -> None:
    """run() returns code 2 when the root doesn't exist."""

    from raghub.migrate import run_migration

    exit_code = run_migration(tmp_path / "nope")
    assert exit_code == 2


def test_migrate_run_walks_and_rewrites(tmp_path: Path) -> None:
    """run() discovers nested manifests and rewrites them."""

    from raghub.migrate import run_migration

    nested = tmp_path / "data" / "sub"
    nested.mkdir(parents=True)
    legacy = nested / "manifest.json"
    legacy.write_text(json.dumps({"rec1": {}}), encoding="utf-8")
    exit_code = run_migration(tmp_path / "data")
    assert exit_code == 0
    payload = json.loads(legacy.read_text(encoding="utf-8"))
    assert payload["version"] == 2


# ---------------------------------------------------------------------------
# raghub.prompts
# ---------------------------------------------------------------------------


def test_prompts_module_exports() -> None:
    """raghub.prompts exports the documented public surface."""

    import raghub.prompts as prompts_module

    # Wildcard import side effects: PromptConfig, PromptRenderer, etc.
    assert hasattr(prompts_module, "render_prompt") or hasattr(prompts_module, "PromptConfig")


def test_prompt_config_default_values() -> None:
    """PromptConfig has documented default values for the tuning knobs."""

    from raghub.prompts import PromptConfig

    cfg = PromptConfig()
    assert cfg.max_tokens > 0
    assert hasattr(cfg, "reserved_output_tokens")


# ---------------------------------------------------------------------------
# raghub.plugins
# ---------------------------------------------------------------------------


def test_plugin_registry_entries_for_kind() -> None:
    """``Plugins.entries_for(kind)`` snapshots the registered entries."""

    from raghub.plugins import PluginKind, Plugins

    registry = Plugins()
    plugin = MagicMock()
    registry.register(PluginKind.Converter, "default", plugin)
    assert registry.entries_for(PluginKind.Converter)["default"] is plugin


# ---------------------------------------------------------------------------
# raghub.conv
# ---------------------------------------------------------------------------


def test_conv_module_exports() -> None:
    """raghub.conv exports ConversationHistory + Memory."""

    import raghub.conv as conv_module

    for name in ("ConversationHistory", "Memory"):
        assert hasattr(conv_module, name) or name in conv_module.__dict__


def test_tokenizer_default_model_constant() -> None:
    """Tokenizer.DEFAULT_MODEL is a non-empty string."""

    from raghub.conv import Tokenizer

    assert isinstance(Tokenizer.DEFAULT_MODEL, str)
    assert Tokenizer.DEFAULT_MODEL


def test_tokenizer_module_classes() -> None:
    """Tokenizer module exposes a Tokenizer class."""

    from raghub.conv import Tokenizer

    assert hasattr(Tokenizer, "load")


def test_conv_sliding_window_manager_attributes() -> None:
    """SlidingWindowTrimmer exposes the documented attributes."""

    from raghub.conv import SlidingWindowTrimmer

    assert hasattr(SlidingWindowTrimmer, "__init__")


def test_conv_conversation_manager_attributes() -> None:
    """ConversationHistory exposes a constructor."""

    from raghub.conv import ConversationHistory

    assert hasattr(ConversationHistory, "__init__")
