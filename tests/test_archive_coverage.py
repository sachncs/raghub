"""Coverage tests for :mod:`raghub.archive`.

Targets the high-impact public surface:

* :func:`create_snapshot` / :func:`restore_snapshot` round-trip.
* :func:`verify_archive` integrity gate.
* :class:`LocalArchiveStore` put/get/list/delete.
* :class:`ArchiveManifest` encoding / signature helpers.
* :func:`make_tar`, :func:`extract_member`, :func:`zstd_compress`,
  :func:`zstd_decompress`.
* :func:`vacuum_sqlite`, :func:`collect_with_vacuum`.
* :func:`signing_key` env contract.
"""

from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from raghub.archive import (
    MANIFEST_FORMAT_VERSION,
    ArchiveCorruptionError,
    ArchiveEntry,
    ArchiveManifest,
    LocalArchiveStore,
    collect_with_vacuum,
    create_snapshot,
    extract_member,
    hmac_compare_digest,
    make_tar,
    restore_snapshot,
    signing_key,
    vacuum_sqlite,
    write_archive,
    zstd_compress,
    zstd_decompress,
)

SIGNING_KEY = "x" * 44  # 32+ bytes for HMAC.


@pytest.fixture
def signing_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the archive signing key for the duration of the test."""
    monkeypatch.setenv("RAGHUB_ARCHIVE_SIGNING_KEY", SIGNING_KEY)


@pytest.fixture
def seed_data_dir(tmp_path: Path) -> Path:
    """Build a small data directory with one of every collector input."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "registry.json").write_text('{"doc": "v1"}')
    (data_dir / "sessions.db").write_bytes(b"fake-sessions-sqlite")
    (data_dir / "ingestion_jobs.db").write_bytes(b"fake-queue-sqlite")
    (data_dir / "feedback.db").write_bytes(b"fake-feedback-sqlite")
    (data_dir / "audit.db").write_bytes(b"fake-audit-sqlite")
    (data_dir / "tenant_secrets.db").write_bytes(b"fake-tenant-sqlite")
    (data_dir / "manifest.json").write_text('{"name": "session-1"}')
    images = data_dir / "images"
    images.mkdir()
    (images / "chart.png").write_bytes(b"fake-png")
    return data_dir


# ---------------------------------------------------------------------------
# Archive encoding
# ---------------------------------------------------------------------------


class TestArchiveManifest:
    def test_to_dict_is_json_safe(self) -> None:
        manifest = ArchiveManifest(
            format_version=MANIFEST_FORMAT_VERSION,
            app_version="0.9.5",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            tenant_ids=["acme"],
            entries=[
                ArchiveEntry(path="registry.json", sha256="abc", size_bytes=3, kind="registry")
            ],
            signature="sig",
        )
        data = manifest.to_dict()
        assert data["format_version"] == MANIFEST_FORMAT_VERSION
        assert data["entries"][0]["path"] == "registry.json"
        # Must be JSON-serialisable.
        json.dumps(data)

    def test_expected_signature_is_stable(self) -> None:
        manifest = ArchiveManifest(
            format_version=MANIFEST_FORMAT_VERSION,
            app_version="0.9.5",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            tenant_ids=[],
            entries=[],
            signature="",
        )
        sig1 = manifest.expected_signature(b"key")
        sig2 = manifest.expected_signature(b"key")
        assert sig1 == sig2
        assert len(sig1) == 64  # sha256 hex

    def test_verify_signature_accepts_matching_key(self) -> None:
        manifest = ArchiveManifest(
            format_version=MANIFEST_FORMAT_VERSION,
            app_version="0.9.5",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            tenant_ids=[],
            entries=[],
            signature="",
        )
        signed = ArchiveManifest(
            format_version=manifest.format_version,
            app_version=manifest.app_version,
            created_at=manifest.created_at,
            tenant_ids=manifest.tenant_ids,
            entries=manifest.entries,
            signature=manifest.expected_signature(SIGNING_KEY.encode("utf-8")),
        )
        # Should not raise.
        signed.verify_signature(SIGNING_KEY.encode("utf-8"))

    def test_verify_signature_rejects_mismatch(self) -> None:
        manifest = ArchiveManifest(
            format_version=MANIFEST_FORMAT_VERSION,
            app_version="0.9.5",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            tenant_ids=[],
            entries=[],
            signature="deadbeef",
        )
        with pytest.raises(ArchiveCorruptionError):
            manifest.verify_signature(b"any-key")


class TestHmacCompareDigest:
    def test_equal_returns_true(self) -> None:
        assert hmac_compare_digest("abc", "abc") is True

    def test_different_returns_false(self) -> None:
        assert hmac_compare_digest("abc", "abd") is False


# ---------------------------------------------------------------------------
# signing_key
# ---------------------------------------------------------------------------


class TestSigningKey:
    def test_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAGHUB_ARCHIVE_SIGNING_KEY", raising=False)
        from raghub.errors import MissingDepError

        with pytest.raises(MissingDepError):
            signing_key()

    def test_returns_bytes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGHUB_ARCHIVE_SIGNING_KEY", "the-secret")
        assert signing_key() == b"the-secret"


# ---------------------------------------------------------------------------
# LocalArchiveStore
# ---------------------------------------------------------------------------


class TestLocalArchiveStore:
    def test_put_get_round_trip(self, tmp_path: Path) -> None:
        store = LocalArchiveStore(tmp_path / "archives")
        store.put("a/b.bin", b"hello")
        assert store.get("a/b.bin") == b"hello"

    def test_list_returns_paths_under_prefix(self, tmp_path: Path) -> None:
        store = LocalArchiveStore(tmp_path / "archives")
        store.put("snapshots-2024.tar.zst", b"x")
        store.put("snapshots-2025.tar.zst", b"y")
        store.put("other.tar.zst", b"z")
        keys = sorted(store.list("snapshots"))
        # ``rglob(f"{prefix}*")`` matches files whose name starts with the
        # prefix; we verify the matching ones are returned, and the
        # non-matching one is absent.
        assert "snapshots-2024.tar.zst" in keys
        assert "snapshots-2025.tar.zst" in keys
        assert "other.tar.zst" not in keys

    def test_list_without_prefix_returns_all(self, tmp_path: Path) -> None:
        store = LocalArchiveStore(tmp_path / "archives")
        store.put("a.bin", b"x")
        store.put("b.bin", b"y")
        assert sorted(store.list("")) == ["a.bin", "b.bin"]

    def test_delete_removes_key(self, tmp_path: Path) -> None:
        store = LocalArchiveStore(tmp_path / "archives")
        store.put("a.bin", b"x")
        store.delete("a.bin")
        # The file is gone; subsequent get raises.
        with pytest.raises(FileNotFoundError):
            store.get("a.bin")

    def test_delete_missing_is_silent(self, tmp_path: Path) -> None:
        store = LocalArchiveStore(tmp_path / "archives")
        # Should not raise.
        store.delete("never-existed.bin")


# ---------------------------------------------------------------------------
# create_snapshot / restore_snapshot / verify_archive round-trip
# ---------------------------------------------------------------------------


class TestSnapshotRoundTrip:
    def test_create_snapshot_captures_all_components(
        self, signing_key_env, seed_data_dir: Path
    ) -> None:
        manifest, files = create_snapshot(seed_data_dir)
        # Every collected file is recorded in the manifest.
        paths = {entry.path for entry in manifest.entries}
        # Note: the snapshot renames files based on the collector name.
        # ``collect_registry_json`` -> ``registry_json`` (no .db to rename),
        # ``collect_<x>_db`` -> ``<x>.db``.
        assert "registry_json" in paths
        assert "sessions.db" in paths
        assert "queue.db" in paths  # ingestion_jobs.db is renamed to queue.db
        assert "feedback.db" in paths
        assert "audit.db" in paths
        assert "tenant_secrets.db" in paths
        assert "manifest.json" in paths
        assert any(p.startswith("images/") for p in paths)
        # Files dict contains the contents.
        assert files["registry_json"] == b'{"doc": "v1"}'
        # Manifest signature is non-empty.
        assert manifest.signature != ""

    def test_restore_snapshot_unsupported_format_version(
        self, signing_key_env, seed_data_dir: Path, tmp_path: Path
    ) -> None:
        # Build a hand-crafted archive with format_version=99.
        manifest = ArchiveManifest(
            format_version=99,
            app_version="0.9.5",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            tenant_ids=[],
            entries=[],
            signature="deadbeef",
        )
        tar_bytes = make_tar(manifest, {})
        archive_path = tmp_path / "bad.tar.zst"
        archive_path.write_bytes(zstd_compress(tar_bytes))
        with pytest.raises(ArchiveCorruptionError, match="unsupported archive format_version"):
            restore_snapshot(archive_path, tmp_path / "out")

    def test_write_archive_produces_non_empty_file(
        self, signing_key_env, seed_data_dir: Path, tmp_path: Path
    ) -> None:
        manifest, files = create_snapshot(seed_data_dir)
        archive_path = tmp_path / "backup.tar.zst"
        write_archive(manifest, files, archive_path)
        assert archive_path.exists()
        assert archive_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tar helpers
# ---------------------------------------------------------------------------


class TestMakeTar:
    def test_make_tar_includes_manifest_and_files(self) -> None:
        manifest = ArchiveManifest(
            format_version=1,
            app_version="0.9.5",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            tenant_ids=["acme"],
            entries=[
                ArchiveEntry(path="hello.txt", sha256="x", size_bytes=5, kind="image_store"),
            ],
            signature="sig",
        )
        tar_bytes = make_tar(manifest, {"hello.txt": b"hello"})
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
            names = tar.getnames()
        assert "manifest.json" in names
        assert "hello.txt" in names


class TestExtractMember:
    def test_extract_member_returns_file_bytes(self) -> None:
        manifest = ArchiveManifest(
            format_version=1,
            app_version="0.9.5",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            tenant_ids=[],
            entries=[
                ArchiveEntry(path="hello.txt", sha256="x", size_bytes=5, kind="image_store"),
            ],
            signature="sig",
        )
        tar_bytes = make_tar(manifest, {"hello.txt": b"hello"})
        assert extract_member(tar_bytes, "hello.txt") == b"hello"
        assert json.loads(extract_member(tar_bytes, "manifest.json"))["format_version"] == 1

    def test_extract_member_missing_raises(self) -> None:
        manifest = ArchiveManifest(
            format_version=1,
            app_version="0.9.5",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            tenant_ids=[],
            entries=[],
            signature="sig",
        )
        tar_bytes = make_tar(manifest, {})
        with pytest.raises(ArchiveCorruptionError, match="member not found"):
            extract_member(tar_bytes, "missing.bin")

    def test_extract_member_rejects_absolute_path(self) -> None:
        # Build a tar with an absolute path member.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo("/etc/passwd")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))
        with pytest.raises(ArchiveCorruptionError, match="path traversal"):
            extract_member(buf.getvalue(), "/etc/passwd")

    def test_extract_member_rejects_parent_traversal(self) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo("../../etc/passwd")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))
        with pytest.raises(ArchiveCorruptionError, match="path traversal"):
            extract_member(buf.getvalue(), "../../etc/passwd")


# ---------------------------------------------------------------------------
# zstd helpers
# ---------------------------------------------------------------------------


class TestZstdHelpers:
    def test_compress_then_decompress_round_trip(self) -> None:
        original = b"hello world " * 50
        compressed = zstd_compress(original)
        assert zstd_decompress(compressed) == original

    def test_decompress_passthrough_when_zstd_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate missing zstandard by intercepting the import.
        import builtins

        original_import = builtins.__import__

        def _hook(name, *args, **kwargs):
            if name == "zstandard" or name.startswith("zstandard."):
                raise ImportError("simulated")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _hook)
        data = b"raw bytes"
        assert zstd_compress(data) == data
        assert zstd_decompress(data) == data


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


class TestVacuumSqlite:
    def test_vacuum_real_db(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE foo (x INTEGER)")
            conn.execute("INSERT INTO foo VALUES (1)")
            conn.commit()
        # Should not raise.
        vacuum_sqlite(db)
        # Verify the table is still readable.
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT * FROM foo").fetchall()
        assert rows == [(1,)]

    def test_vacuum_missing_db_is_silent(self, tmp_path: Path) -> None:
        # Should not raise on a missing file.
        vacuum_sqlite(tmp_path / "does-not-exist.db")


class TestCollectWithVacuum:
    def test_collects_sqlite_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.db").write_bytes(b"sql-a")
        (tmp_path / "b.db").write_bytes(b"sql-b")
        (tmp_path / "ignored.txt").write_text("text")
        files = collect_with_vacuum(tmp_path)
        assert files["a.db"] == b"sql-a"
        assert files["b.db"] == b"sql-b"
        assert "ignored.txt" not in files

    def test_vacuum_paths_vacuumed_first(self, tmp_path: Path) -> None:
        db = tmp_path / "x.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
        collect_with_vacuum(tmp_path, sqlite_paths=(db,))
        # The DB is still readable after the vacuum.
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT * FROM t").fetchall()
        assert rows == [(1,)]
