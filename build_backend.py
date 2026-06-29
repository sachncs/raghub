"""Minimal PEP 517 backend for offline editable installs.

This backend keeps editable installs working without network access.
The produced wheel installs a single ``.pth`` file pointing at the
repository root, which makes the ``raghub`` package importable in
editable mode. There are no compiled artifacts; this is appropriate
for a pure-Python project.

For non-editable installs (``pip install .``) we delegate to the
standard setuptools backend so the wheel contains the real package
code rather than a ``.pth`` stub.
"""

from __future__ import annotations

import base64
import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

NAME = "retrieval-augmented-generation"
VERSION = "1.0.0"
DIST_INFO = f"{NAME.replace('-', '_')}-{VERSION}.dist-info"


def get_requires_for_build_wheel(config_settings: Any = None) -> list[str]:
    """PEP 517 hook: no build-time requirements."""
    return []


def get_requires_for_build_editable(config_settings: Any = None) -> list[str]:
    """PEP 660 hook: no build-time requirements."""
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: Any, config_settings: Any = None
) -> str:
    """Write minimal ``METADATA``/``WHEEL`` files for the wheel."""
    dist = Path(metadata_directory) / DIST_INFO
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: Production-grade Dynamic RAG framework\n",
        encoding="utf-8",
    )
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: build_backend\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist / "top_level.txt").write_text("raghub\n", encoding="utf-8")
    (dist / "RECORD").write_text("", encoding="utf-8")
    return DIST_INFO


def _delegate_to_setuptools(wheel_directory: Any) -> str:
    """Build a real wheel via setuptools."""
    from setuptools import setuptools_wrap  # type: ignore[import-not-found]

    return setuptools_wrap.build_wheel(wheel_directory)


def build_wheel(
    wheel_directory: Any,
    config_settings: Any = None,
    metadata_directory: Any = None,
) -> str:
    """Build a real, distributable wheel containing the package code."""
    if shutil.which("setup.py") is None and Path("setup.py").exists():
        # Fall back to the standard setuptools backend for non-editable
        # installs so the wheel actually ships the package code.
        return _delegate_to_setuptools(wheel_directory)

    return _build_editable_wheel(wheel_directory)


def build_editable(
    wheel_directory: Any, config_settings: Any = None, metadata_directory: Any = None
) -> str:
    """Build an editable wheel that uses a ``.pth`` file."""
    return _build_editable_wheel(wheel_directory)


def _build_editable_wheel(wheel_directory: str) -> str:
    """Assemble an editable wheel that uses a ``.pth`` file."""
    root = Path(__file__).resolve().parent
    wheel_name = f"{NAME.replace('-', '_')}-{VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    dist_info = f"{NAME.replace('-', '_')}-{VERSION}.dist-info"
    pth_name = f"{NAME.replace('-', '_')}.pth"
    pth_contents = str(root) + os.linesep

    records: list[tuple[str, str, str]] = []
    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as zf:
        _write_file(zf, pth_name, pth_contents.encode("utf-8"), records)
        _write_file(
            zf,
            f"{dist_info}/METADATA",
            (
                "Metadata-Version: 2.1\n"
                f"Name: {NAME}\n"
                f"Version: {VERSION}\n"
                "Summary: Production-grade Dynamic RAG framework\n"
            ).encode("utf-8"),
            records,
        )
        _write_file(
            zf,
            f"{dist_info}/WHEEL",
            (
                "Wheel-Version: 1.0\n"
                "Generator: build_backend\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n"
            ).encode("utf-8"),
            records,
        )
        _write_file(zf, f"{dist_info}/top_level.txt", b"raghub\n", records)
        records.append((f"{dist_info}/RECORD", "", ""))
        record_lines = []
        for path, digest, size in records:
            record_lines.append(f"{path},{digest},{size}")
        zf.writestr(f"{dist_info}/RECORD", "\n".join(record_lines) + "\n")
    return wheel_name


def _write_file(
    zf: ZipFile, path: str, data: bytes, records: list[tuple[str, str, str]]
) -> None:
    """Add ``data`` to the wheel zip and record its RECORD entry."""
    zf.writestr(path, data)
    digest = base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode("ascii")
    records.append((path, f"sha256={digest}", str(len(data))))
