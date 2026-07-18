#!/usr/bin/env python3
"""Build a fully-locked production requirements file for raghub.

Runs ``pip-compile`` against ``pyproject.toml`` and writes the result
to ``requirements-lock.txt`` at the repository root. The locked file
is what production images and the ``pip-audit --strict`` CI gate
consume; the loose ``pyproject.toml`` ranges stay for developers.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Compile a locked requirements file and write it to the repo root.

    Returns:
        The pip-compile exit code.
    """
    repo_root = Path(__file__).resolve().parents[1]
    output = repo_root / "requirements-lock.txt"
    cmd = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        "--quiet",
        "--upgrade",
        "--resolver=backtracking",
        "--output-file",
        str(output),
        str(repo_root / "pyproject.toml"),
        "-c",
        "/tmp/req-overrides.in",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
