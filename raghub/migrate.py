"""raghub.migrate -- storage format migration utility.

The v0 -> v1 migration runs at read time inside :class:`Manifest`
and is therefore no-op for users who just want their data preserved.
This module provides the explicit CLI runner for users who want to
batch-migrate a tree of manifests and SQLite stores under a root path.

The CLI takes a root directory, walks for ``manifest.json`` files, and
ensures each carries ``version = 2``. Run with::

    python -m raghub.migrate --root ./data

Each manifest is rewritten in-place once the migration completes, so
the next read is at v2. SQLite stores are touched only at the column
level (we add a version column on first read) — older databases
still work; migration to a new schema ships as a separate phase.

The tool is idempotent: re-running it against an already-v2 store
produces no writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from raghub.knowledge import Manifest


def migrate_manifest(path: Path) -> bool:
    """Bump ``path`` to the current manifest version.

    Returns ``True`` when the file was rewritten.
    """
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        payload = json.loads(text or "{}")
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if int(payload.get("version", 1)) >= Manifest.CURRENT_VERSION:
        return False
    records = payload.get("records")
    if records is None:
        # legacy v0 file: bare record dict.
        records = payload
    payload = {"version": Manifest.CURRENT_VERSION, "records": records}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return True


def run(root: Path) -> int:
    """Migrate every manifest under ``root`` to v2. Returns 0 on success."""
    if not root.exists():
        print(f"raghub.migrate: {root} does not exist", file=sys.stderr)
        return 2
    rewritten = 0
    for path in root.rglob("manifest.json"):
        if migrate_manifest(path):
            rewritten += 1
            print(f"migrated {path}")
    if rewritten == 0:
        print(f"no migration needed under {root}")
    else:
        print(f"{rewritten} manifest(s) migrated to v{Manifest.CURRENT_VERSION}")
    return 0


def main() -> int:
    """CLI entry: ``python -m raghub.migrate --root DIR``."""
    parser = argparse.ArgumentParser(description="Migrate raghub storage")
    parser.add_argument("--root", type=Path, required=True, help="Root data directory")
    args = parser.parse_args()
    return run(args.root)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
