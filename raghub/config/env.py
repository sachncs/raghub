"""Environment parsing and profile file reading.

Exposes the primitive coercion helpers (:func:`env_bool`,
:func:`env_int`, :func:`env_float`, :func:`csv_to_transforms`) and the
profile-file discovery logic (:func:`load_profile` and friends) that the
loader functions compose.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from raghub.constants import ENV_RAG_PROFILE

__all__ = [
    "TRANSFORM_NAMES",
    "TRUTHY",
    "TransformName",
    "csv_to_transforms",
    "env_bool",
    "env_float",
    "env_int",
    "load_profile",
    "read_toml_file",
    "resolve_config_dir",
]

TRUTHY = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool) -> bool:
    """Return the boolean value of ``os.getenv(name, ...)``.

    Treats ``"1"`` / ``"true"`` / ``"yes"`` / ``"on"`` (case-insensitive)
    as ``True``. Any other non-empty value is ``False``. Missing var
    falls back to ``default``.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY


def env_int(name: str, default: int) -> int:
    """Read ``name`` from the environment as an int.

    Args:
        name: Environment variable name.
        default: Value when the env var is unset.

    Returns:
        Parsed integer, or ``default``.

    Raises:
        ConfigurationError: When the env var is set but not parseable
            as an int.

    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        from raghub.errors import ConfigurationError

        raise ConfigurationError(f"{name}={raw!r} is not a valid integer") from exc


def env_float(name: str, default: float) -> float:
    """Read ``name`` from the environment as a float.

    Args:
        name: Environment variable name.
        default: Value when the env var is unset.

    Returns:
        Parsed float, or ``default``.

    Raises:
        ConfigurationError: When the env var is set but not parseable
            as a float.

    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        from raghub.errors import ConfigurationError

        raise ConfigurationError(f"{name}={raw!r} is not a valid float") from exc


TransformName = Literal["hyde", "multi_query", "step_back", "decompose"]
TRANSFORM_NAMES: tuple[TransformName, ...] = ("hyde", "multi_query", "step_back", "decompose")


def csv_to_transforms(raw: str, default: list[str]) -> list[TransformName]:
    """Parse a comma-separated env var into a validated transform list.

    Unknown names are dropped silently — config files are validated by
    Pydantic and raise on bad values; the env path is forgiving so a
    typo doesn't prevent startup.
    """
    if not raw:
        return cast(
            list[TransformName],
            [name for name in default if name in TRANSFORM_NAMES],
        )
    out: list[TransformName] = []
    for chunk in raw.split(","):
        name = chunk.strip().lower()
        if name and name in TRANSFORM_NAMES and name not in out:
            out.append(name)
    return out


def read_toml_file(path: Path) -> dict[str, Any]:
    """Load a TOML file using :mod:`tomllib` (3.11+).

    Args:
        path: Path to the TOML file.

    Returns:
        The parsed dict, or ``{}`` if the file is empty.

    Raises:
        FileNotFoundError: When ``path`` doesn't exist.
        tomllib.TOMLDecodeError: When the TOML is malformed.
        OSError: When the file cannot be read.

    """
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8")) or {}


def load_profile(profile: str | None) -> tuple[str | None, Path, dict[str, Any]]:
    """Read the YAML + TOML profile files into a single payload dict.

    Search order for the profile directory:

    1. ``RAG_CONFIG_DIR`` environment variable.
    2. ``./config`` relative to the current working directory.
    3. ``~/.config/raghub`` (XDG user config dir).
    4. The bundled ``config/`` shipped with the package.

    Returns:
        Tuple of (selected_profile, profile_path, payload). The
        ``profile_path`` is the YAML path searched; it may not exist.
        Missing files simply contribute an empty payload.

    """
    base_dir = resolve_config_dir()
    selected_profile = profile or os.getenv(ENV_RAG_PROFILE, "development")
    profile_path = base_dir / f"{selected_profile}.yaml"
    toml_path = base_dir / f"{selected_profile}.toml"
    payload: dict[str, Any] = {}
    if profile_path.exists():
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if toml_path.exists():
        toml_payload = read_toml_file(toml_path)
        if toml_payload:
            # TOML takes precedence over YAML when both are present.
            payload = {**payload, **toml_payload}
    return selected_profile, profile_path, payload


def resolve_config_dir() -> Path:
    """Return the directory to search for profile YAML/TOML files.

    Resolution order:
    1. ``RAG_CONFIG_DIR`` env var (explicit override).
    2. ``./config`` (CWD-relative).
    3. ``~/.config/raghub`` (XDG-style user config).
    4. ``config/`` shipped with the package (read-only default).
    """
    env = os.getenv("RAG_CONFIG_DIR")
    if env:
        return Path(env)
    cwd_dir = Path.cwd() / "config"
    if cwd_dir.is_dir():
        return cwd_dir
    xdg_dir = Path.home() / ".config" / "raghub"
    if xdg_dir.is_dir():
        return xdg_dir
    try:
        from importlib.resources import files

        bundled = files("raghub").joinpath("config")
        if bundled.is_dir() and (bundled / "development.yaml").exists():
            return Path(str(bundled))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    # Final fallback: return the CWD-relative path even if it
    # doesn't exist yet — the caller treats missing files as no-ops.
    return cwd_dir
