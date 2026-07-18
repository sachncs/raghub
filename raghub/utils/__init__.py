"""Utility helpers.

This package ships small, dependency-free helpers used across the
codebase:

* :func:`atomic_write_json` — atomic disk write via a temporary file
  and ``os.replace``.
* :func:`load_json` — JSON loader with a sensible default.
* :func:`retry` (in :mod:`.retry`) — exponential-backoff retry for
  transient upstream errors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

__all__ = ["atomic_write_json", "load_json"]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON to ``path``.

    The write goes through a sibling temp file followed by
    :func:`os.replace`, which is atomic on POSIX and Windows. This
    prevents readers from observing a partially-written file even
    when the process is killed mid-write.

    Args:
        path: Destination path. Parent directories are created
            automatically.
        payload: The dict to serialize. ``default=str`` is passed to
            :func:`json.dump` so non-JSON-native values fall back to
            their string representation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temp_name = handle.name
    os.replace(temp_name, path)


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load JSON from ``path``, returning ``default`` when missing.

    Args:
        path: Path to the JSON file.
        default: Value returned when ``path`` does not exist. When
            ``None``, an empty dict is returned.

    Returns:
        The parsed dict, or ``default`` when the file is missing.
        Parse errors propagate as :class:`json.JSONDecodeError`.
    """
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
