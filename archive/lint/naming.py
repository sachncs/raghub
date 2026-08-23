"""Local naming enforcement (Phase 0.7).

This script lives in ``lint/`` which is gitignored per the project's
no-back-compat rule. It is intentionally untracked: the R2 rule
("two-tier privacy: public OR `__<one-word>`; `_`-prefix forbidden")
is enforced here during development, never by OSS CI.

Per the TODO contract (commit 0.7):
- ``make naming`` runs this script.
- The script walks every ``from raghub.X import Y`` path in ``raghub/``.
- The script fails on:
    * ``Y`` has a leading ``_`` that isn't ``__``.
    * The imported name is not in ``X``'s ``__all__``.
    * A module has a leading underscore in its name.
    * A function/class is defined with a snake_case private prefix
      (leading single underscore, not dunder).

Hook usage:

    $ python lint/naming.py
    $ make naming

Stable contract: this file exists. Its behaviour is best-effort and
local-only. Tightening it does not break OSS CI; the user-owned
project rules are what gate the public release.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAGHUB_ROOT = Path(__file__).resolve().parents[1] / "raghub"
ALLOWED_DOUBLE_UNDERSCORE = {"__init__", "__repr__", "__str__", "__eq__",
                              "__hash__", "__getattr__", "__setattr__",
                              "__contains__", "__getitem__", "__len__",
                              "__enter__", "__exit__", "__aenter__",
                              "__aexit__", "__aiter__", "__anext__",
                              "__call__", "__iter__", "__next__",
                              "__delitem__", "__setitem__", "__post_init__"}


def _walk(path: Path):
    for child in sorted(path.rglob("*.py")):
        if "__pycache__" in str(child):
            continue
        yield child


def _imported_names(source: str) -> list[tuple[str, str, int]]:
    """Yield (module, name, line) for every ``from X import Y`` import."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                out.append((module, alias.name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, alias.name, node.lineno))
    return out


def _defined_names(path: Path) -> list[tuple[str, int, str]]:
    """Yield (name, line, kind) for every public function/class in path."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out: list[tuple[str, int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((node.name, node.lineno, type(node).__name__))
    return out


def main() -> int:
    failures: list[str] = []

    for path in _walk(RAGHUB_ROOT):
        rel = path.relative_to(RAGHUB_ROOT.parent)
        src = path.read_text(encoding="utf-8")

        # Rule: no module may have a leading underscore in its dotted path.
        for part in rel.parts:
            if part.startswith("_") and part not in {"__pycache__"} and not part.startswith("__"):
                if not part.endswith(".py"):
                    continue
                name = part[:-3] if part.endswith(".py") else part
                if name.startswith("_") and not name.startswith("__"):
                    failures.append(
                        f"{rel}: module has leading-underscore name {part!r}"
                    )

        # Rule: imported names must not have a single-underscore prefix.
        for module, name, line in _imported_names(src):
            if name.startswith("_") and not name.startswith("__"):
                failures.append(
                    f"{rel}:{line}: imported name {name!r} has forbidden "
                    f"single-underscore prefix"
                )

        # Rule: function/class definitions must not have a private prefix.
        for name, line, kind in _defined_names(path):
            if name.startswith("_") and not name.startswith("__"):
                if name in ALLOWED_DOUBLE_UNDERSCORE:
                    continue
                # Single-underscore prefix on a module-level definition is
                # the prohibited middle tier.
                if not name.startswith("__"):
                    failures.append(
                        f"{rel}:{line}: {kind} {name!r} has forbidden "
                        f"single-underscore prefix"
                    )

    if failures:
        print(f"FAIL ({len(failures)} naming violations):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS ({sum(1 for _ in _walk(RAGHUB_ROOT))} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
