"""Minimal PEP 517 backend for offline editable installs.

This backend keeps editable installs working without network access.

The produced wheel installs a single ``.pth`` file pointing at the
repository root, which makes the ``raghub`` package importable in
editable mode. There are no compiled artifacts; this is appropriate
for a pure-Python project.
"""

from __future__ import annotations

import base64
import os
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

NAME = "retrieval-augmented-generation"
VERSION = "1.0.0"
DIST_INFO = f"{NAME.replace('-', '_')}-{VERSION}.dist-info"


def get_requires_for_build_wheel(config_settings=None):  # noqa: D401
    """PEP 517 hook: no build-time requirements."""
    return []


def get_requires_for_build_editable(config_settings=None):  # noqa: D401
    """PEP 660 hook: no build-time requirements."""
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Write minimal ``METADATA``/``WHEEL`` files for the wheel.

    Args:
        metadata_directory: Target directory for the dist-info.
        config_settings: Ignored.

    Returns:
        The dist-info directory name.
    """
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
    (dist / "top_level.txt").write_text("raghub\napp\n", encoding="utf-8")
    (dist / "RECORD").write_text("", encoding="utf-8")
    return DIST_INFO


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a regular (non-editable) wheel.

    Delegates to :func:`_build_editable_wheel` because the project is
    pure-Python and the result is identical.

    Args:
        wheel_directory: Target directory for the wheel.
        config_settings: Ignored.
        metadata_directory: Ignored.

    Returns:
        The wheel filename.
    """
    return _build_editable_wheel(wheel_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable wheel.

    Args:
        wheel_directory: Target directory for the wheel.
        config_settings: Ignored.
        metadata_directory: Ignored.

    Returns:
        The wheel filename.
    """
    return _build_editable_wheel(wheel_directory)


def _build_editable_wheel(wheel_directory: str) -> str:
    """Assemble an editable wheel that uses a ``.pth`` file.

    Args:
        wheel_directory: Target directory for the wheel.

    Returns:
        The wheel filename.
    """
    root = Path(__file__).resolve().parent
    wheel_name = f"{NAME.replace('-', '_')}-{VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    dist_info = f"{NAME.replace('-', '_')}-{VERSION}.dist-info"
    pth_name = f"{NAME.replace('-', '_')}.pth"
    # ``.pth`` files add their directory to ``sys.path`` at
    # interpreter startup; this is how editable installs work.
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
        _write_file(zf, f"{dist_info}/top_level.txt", b"raghub\napp\n", records)
        records.append((f"{dist_info}/RECORD", "", ""))
        record_lines = []
        for path, digest, size in records:
            record_lines.append(f"{path},{digest},{size}")
        zf.writestr(f"{dist_info}/RECORD", "\n".join(record_lines) + "\n")
    return wheel_name


def _write_file(
    zf: ZipFile, path: str, data: bytes, records: list[tuple[str, str, str]]
) -> None:
    """Add ``data`` to the wheel zip and record its RECORD entry.

    Args:
        zf: The wheel :class:`ZipFile`.
        path: Path inside the wheel.
        data: File contents.
        records: Mutable list of RECORD triples; appended to here.
    """
    zf.writestr(path, data)
    digest = base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode("ascii")
    records.append((path, f"sha256={digest}", str(len(data))))