"""Common CLI helpers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from raghub.config.settings import AppSettings


def load_settings_or_path(path: str | None) -> "AppSettings":
    """Load settings from a config file path or the active profile.

    Args:
        path: Optional path to a YAML or TOML profile. When ``None``,
            :func:`raghub.config.settings.load_settings` is used
            with the active profile.

    Returns:
        The loaded :class:`AppSettings`.
    """
    from raghub.config.settings import AppSettings, load_settings

    if path is None:
        return load_settings()
    import yaml

    if str(path).endswith(".toml"):
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        data = tomllib.loads(Path(path).read_text(encoding="utf-8")) or {}
    else:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppSettings(**{k: v for k, v in data.items() if k in AppSettings.model_fields})


def print_json(payload: Any) -> None:
    """Print ``payload`` as pretty JSON to stdout."""
    print(json.dumps(payload, indent=2, default=str))


def run_async(coro: Any) -> Any:
    """Run an async coroutine and return its result."""
    return asyncio.run(coro)


__all__ = ["load_settings_or_path", "print_json", "run_async"]
