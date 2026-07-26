"""Shared CLI formatting helpers — JSON I/O + top-level imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from raghub.config import Settings
from raghub.rag import RAG
from raghub.utils import write_json as write_json_impl


def write_json(payload: Any) -> None:
    """Write ``payload`` as pretty JSON to stdout.

    Args:
        payload: Any JSON-serialisable value.
    """
    write_json_impl(payload)


def read_settings(path: str | None) -> Any:
    """Load :class:`raghub.config.Settings` from ``path`` or the active profile.

    Args:
        path: Optional YAML/TOML path. ``None`` calls :meth:`Settings.load`.

    Returns:
        The parsed :class:`Settings`.
    """
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
    return RAG.from_config(config) if config else RAG()


