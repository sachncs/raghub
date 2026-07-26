from __future__ import annotations

"""RAGHub CLI.

Single Typer-based CLI with all commands attached as sub-apps or
top-level commands in :mod:`raghub.cli.main`.

Public surface: ``from raghub.cli import app, main``.
"""

from raghub.cli.main import app, main

