"""Shared CLI formatting helpers — JSON I/O + top-level imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


def write_json(payload: Any) -> None:
    """Write ``payload`` as pretty JSON to stdout.

    Args:
        payload: Any JSON-serialisable value.
    """
    sys.stdout.write(json.dumps(payload, indent=2, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def read_settings(path: str | None) -> Any:
    """Load :class:`raghub.config.Settings` from ``path`` or the active profile.

    Args:
        path: Optional YAML/TOML path. ``None`` calls :meth:`Settings.load`.

    Returns:
        The parsed :class:`Settings`.
    """
    from raghub.config import Settings

    if path is None:
        return Settings.load()
    if path.endswith(".toml"):
        import tomllib

        data = tomllib.loads(Path(path).read_text(encoding="utf-8")) or {}
    else:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings(**{k: v for k, v in data.items() if k in Settings.model_fields})


def make_rag(config: str | None) -> Any:
    """Instantiate a :class:`raghub.RAG` from ``config`` or defaults.

    Args:
        config: Optional path to a YAML/TOML config file.

    Returns:
        A configured :class:`raghub.RAG` instance.
    """
    from raghub.rag import RAG

    return RAG.from_config(config) if config else RAG()


__all__ = ["make_rag", "read_settings", "write_json"]
