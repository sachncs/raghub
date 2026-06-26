"""Minimal PEP 517 backend for offline editable installs.

This backend keeps editable installs working without network access.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import base64
import os
from zipfile import ZIP_DEFLATED, ZipFile


NAME = "retrieval-augmented-generation"
VERSION = "1.0.0"
DIST_INFO = f"{NAME.replace('-', '_')}-{VERSION}.dist-info"


def get_requires_for_build_wheel(config_settings=None):  # noqa: D401
    return []


def get_requires_for_build_editable(config_settings=None):  # noqa: D401
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
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
    return _build_editable_wheel(wheel_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_editable_wheel(wheel_directory)


def _build_editable_wheel(wheel_directory: str) -> str:
    root = Path(__file__).resolve().parent
    src_path = root / "src"
    wheel_name = f"{NAME.replace('-', '_')}-{VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    dist_info = f"{NAME.replace('-', '_')}-{VERSION}.dist-info"
    pth_name = f"{NAME.replace('-', '_')}.pth"
    pth_contents = str(root) + os.linesep + str(src_path) + os.linesep

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


def _write_file(zf: ZipFile, path: str, data: bytes, records: list[tuple[str, str, str]]) -> None:
    zf.writestr(path, data)
    digest = base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode("ascii")
    records.append((path, f"sha256={digest}", str(len(data))))
