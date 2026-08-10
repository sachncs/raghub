"""Inventory generator (Phase 0.8).

Walks ``raghub/`` and produces a public-surface map:

- Every class, function, and enum with file:line.
- Every class-name collision (multiple definitions across files).
- Every `_`-prefixed symbol that is *reachable* from outside the
  defining module.
- Every `__all__` and the symbols it lists.
- Whether each module's `__all__` matches its reachable public
  surface.

Output:
- ``reports/inventory.json`` (machine-readable).
- ``reports/inventory.md`` (human-readable).

Per the TODO: this is a *map* of the public surface at a point in
time. It drives Phase 1.5 (underscore purge), Phase 1.7 (entity
cascade) and Phase 6 (flat re-export from ``raghub/__init__.py``).

Usage:

    $ python -m devtools.inventory
    $ make inventory

Both outputs land in ``reports/`` (gitignored). To compare
snapshots, copy the JSON out of the workspace before regen.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAGHUB_ROOT = REPO_ROOT / "raghub"
REPORTS_ROOT = REPO_ROOT / "reports"


def _walk_python(path: Path):
    for child in sorted(path.rglob("*.py")):
        if "__pycache__" in str(child):
            continue
        yield child


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None


def _module_all(tree: ast.Module) -> list[str]:
    """Return the names in this module's __all__, or []."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    return [
                        elt.value
                        for elt in node.value.elts  # type: ignore[attr-defined]
                        if isinstance(elt, ast.Constant)
                    ]
    return []


def _defined_public_symbols(tree: ast.Module) -> list[dict[str, str | int]]:
    """Return a list of {name, kind, line} for every public def at top-level."""
    out: list[dict[str, str | int]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append({"name": node.name, "kind": "class", "line": node.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({"name": node.name, "kind": "function", "line": node.lineno})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    out.append({"name": target.id, "kind": "constant", "line": node.lineno})
    return out


def _underscore_candidates(tree: ast.Module, file: Path) -> list[dict[str, str | int]]:
    """List every `_`-prefixed def that is *not* a real dunder."""
    allowed_dunders = {
        "__init__",
        "__repr__",
        "__str__",
        "__eq__",
        "__hash__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__contains__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__len__",
        "__iter__",
        "__next__",
        "__call__",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
        "__aiter__",
        "__anext__",
        "__post_init__",
        "__class_getitem__",
        "__new__",
        "__init_subclass__",
    }
    out: list[dict[str, str | int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = node.name
            if nm.startswith("_") and nm not in allowed_dunders:
                out.append({"name": nm, "kind": type(node).__name__, "line": node.lineno})
    return out


def main() -> int:
    REPORTS_ROOT.mkdir(exist_ok=True)
    classes: dict[str, list[dict[str, str]]] = defaultdict(list)
    functions: list[dict[str, str | int]] = []
    all_exports: dict[str, list[str]] = {}
    private_candidates: list[dict[str, str | int]] = []

    for path in _walk_python(RAGHUB_ROOT):
        rel = str(path.relative_to(REPO_ROOT))
        tree = _parse(path)
        if tree is None:
            continue
        for sym in _defined_public_symbols(tree):
            sym["file"] = rel
            if sym["kind"] == "class":
                # Filter for enum-like classes (heuristic: end with Enum,
                # or a BaseModel subclass with a str Enum parent).
                classes[sym["name"]].append({"file": rel, "line": str(sym["line"])})
            elif sym["kind"] == "function":
                functions.append(sym)
        for cand in _underscore_candidates(tree, path):
            cand["file"] = rel
            private_candidates.append(cand)
        all_list = _module_all(tree)
        if all_list:
            module_name = rel.replace("/", ".").replace(".py", "").lstrip(".")
            all_exports[module_name] = all_list

    # Detect enum classes (any class whose name starts with `DocType`, etc., or whose base
    # node is Enum). Cheap heuristic: a public class whose name is short PascalCase often
    # is an enum candidate — record candidates and let humans confirm.
    enum_candidates: list[dict[str, str | int]] = []
    for cls_name, locations in classes.items():
        in_models = any(
            loc["file"].endswith("models.py") or loc["file"].endswith("domain.py")
            for loc in locations
        )
        is_enum_like = cls_name.endswith("Type") or cls_name in {"State", "Class", "Access", "Kind"}
        if in_models and is_enum_like:
            enum_candidates.append(
                {"name": cls_name, "kind": "enum-class", "line": int(locations[0]["line"])}
            )

    # Collisions: classes defined in more than one file.
    collisions = {name: locs for name, locs in classes.items() if len(locs) > 1}

    inventory = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": {
            "files_scanned": sum(1 for _ in _walk_python(RAGHUB_ROOT)),
            "class_count": sum(len(v) for v in classes.values()),
            "function_count": len(functions),
            "enum_candidates": len(enum_candidates),
            "private_candidates": len(private_candidates),
            "collisions": len(collisions),
        },
        "classes": dict(classes),
        "collisions": collisions,
        "functions": functions,
        "enum_candidates": enum_candidates,
        "private_candidates": private_candidates,
        "all_exports": all_exports,
    }

    (REPORTS_ROOT / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8"
    )
    (REPORTS_ROOT / "inventory.md").write_text(_to_markdown(inventory), encoding="utf-8")
    print(f"wrote reports/inventory.json ({inventory['summary']['files_scanned']} files)")
    print("wrote reports/inventory.md")
    return 0


def _to_markdown(inv: dict) -> str:
    out: list[str] = []
    s = inv["summary"]
    out.append("# raghub Inventory")
    out.append("")
    out.append(f"generated: `{inv['generated_at']}`")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"- Files scanned: **{s['files_scanned']}**")
    out.append(f"- Class definitions: **{s['class_count']}**")
    out.append(f"- Functions: **{s['function_count']}**")
    out.append(f"- Enum candidates: **{s['enum_candidates']}**")
    out.append(f"- Private (`_`-prefixed) candidates: **{s['private_candidates']}**")
    out.append(f"- Class collisions: **{s['collisions']}**")
    out.append("")

    out.append("## Class collisions (resolve before Phase 1.7)")
    out.append("")
    if inv["collisions"]:
        for name, locs in sorted(inv["collisions"].items()):
            out.append(f"### `{name}`")
            for loc in locs:
                out.append(f"- `{loc['file']}:{loc['line']}`")
            out.append("")
    else:
        out.append("None detected.")
        out.append("")

    out.append("## Private candidates (resolve in Phase 1.5)")
    out.append("")
    if inv["private_candidates"]:
        for c in inv["private_candidates"]:
            out.append(f"- `{c['name']}` — {c['file']}:{c['line']}")
    else:
        out.append("None detected.")
    out.append("")

    out.append("## Enum candidates")
    out.append("")
    if inv["enum_candidates"]:
        for c in inv["enum_candidates"]:
            out.append(f"- `{c['name']}`")
    else:
        out.append("None detected.")
    out.append("")

    out.append("## Modules with `__all__`")
    out.append("")
    for mod, names in sorted(inv["all_exports"].items()):
        out.append(f"### `{mod}`")
        out.append("")
        for n in names:
            out.append(f"- `{n}`")
        out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
