"""Tests for the :class:`JsonDocumentRegistry` storage helper."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from raghub.exceptions import StorageError
from raghub.models import Classification, DocumentLifecycleStatus, DocumentRecord, Visibility
from raghub.storage.json_registry import JsonDocumentRegistry


def _make_record(**overrides) -> DocumentRecord:
    base = dict(
        document_id="doc-1",
        version=1,
        checksum="abc",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        owner="alice@x",
        organization="Apple",
        department="",
        tags=[],
        classification=Classification.INTERNAL,
        visibility=Visibility.ORGANIZATION,
        status=DocumentLifecycleStatus.READY,
        filename="report.pdf",
        file_type="pdf",
        mime_type="application/pdf",
        chunk_count=0,
        chunk_ids=[],
    )
    base.update(overrides)
    return DocumentRecord(**base)


def test_registry_load_missing_file(tmp_path: Path) -> None:
    """Loading a non-existent registry yields empty state."""
    reg = JsonDocumentRegistry(tmp_path / "missing.json")
    assert reg.documents == {}
    assert reg.checksum_index == {}


def test_registry_load_corrupt_file(tmp_path: Path) -> None:
    """A corrupt file is tolerated; the registry starts empty."""
    p = tmp_path / "reg.json"
    p.write_text("not json at all", encoding="utf-8")
    reg = JsonDocumentRegistry(p)
    assert reg.documents == {}


def test_registry_save_round_trip(tmp_path: Path) -> None:
    """Saving then loading the registry preserves every record."""
    p = tmp_path / "reg.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record())
    fresh = JsonDocumentRegistry(p)
    assert "doc-1" in fresh.documents
    assert fresh.get_latest("doc-1").owner == "alice@x"


def test_registry_get_latest_unknown_returns_none(tmp_path: Path) -> None:
    """Asking for an unknown document returns ``None``."""
    reg = JsonDocumentRegistry(tmp_path / "r.json")
    assert reg.get_latest("missing") is None


def test_registry_get_specific_version(tmp_path: Path) -> None:
    """A specific version can be retrieved after multiple writes."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record(version=1))
    reg.save_version(_make_record(version=2))
    reg.save_version(_make_record(version=3))
    assert reg.get_specific_version("doc-1", 2).version == 2
    assert reg.get_specific_version("doc-1", 99) is None


def test_registry_save_replaces_in_place(tmp_path: Path) -> None:
    """Re-saving the same version number replaces the existing record."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record(version=1, filename="a.pdf"))
    reg.save_version(_make_record(version=1, filename="b.pdf"))
    assert reg.get_latest("doc-1").filename == "b.pdf"
    # Only one version remains.
    assert len(reg.documents["doc-1"]) == 1


def test_registry_newer_version_archives_prior(tmp_path: Path) -> None:
    """Writing a newer version archives the prior latest."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record(version=1))
    reg.save_version(_make_record(version=2))
    assert reg.get_specific_version("doc-1", 1).status == DocumentLifecycleStatus.ARCHIVED


def test_registry_get_by_checksum(tmp_path: Path) -> None:
    """The checksum index resolves a checksum to its document id and version."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record(checksum="cafebabe"))
    rec = reg.get_by_checksum("cafebabe")
    assert rec is not None
    assert rec.document_id == "doc-1"
    assert reg.get_by_checksum("nope") is None


def test_registry_list_accessible_filters_company_and_archived(tmp_path: Path) -> None:
    """``list_accessible`` returns only latest, non-archived, matching-company docs."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record(document_id="a", organization="Apple"))
    reg.save_version(_make_record(document_id="b", organization="Microsoft"))
    reg.save_version(_make_record(document_id="c", organization="Apple", version=1))
    reg.save_version(_make_record(document_id="c", organization="Apple", version=2))
    out = reg.list_accessible(["Apple"])
    assert {d.document_id for d in out} == {"a", "c"}
    assert (
        reg.list_accessible(["Microsoft"])
        and reg.list_accessible(["Microsoft"])[0].document_id == "b"
    )
    assert reg.list_accessible(["Amazon"]) == []


def test_registry_archive_unknown_is_noop(tmp_path: Path) -> None:
    """Archiving a missing document is a no-op (no exception)."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.archive("nope")
    assert reg.documents == {}


def test_registry_archive_marks_latest_archived(tmp_path: Path) -> None:
    """``archive`` flips the latest version's status."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record())
    reg.archive("doc-1")
    assert reg.get_latest("doc-1").status == DocumentLifecycleStatus.ARCHIVED


def test_registry_dump_returns_snapshot(tmp_path: Path) -> None:
    """``dump`` returns an in-memory snapshot of the registry."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    reg.save_version(_make_record())
    snap = reg.dump()
    assert "doc-1" in snap.documents
    assert snap.checksum_index["abc"] == ("doc-1", 1)


def test_registry_save_propagates_storage_error(tmp_path: Path, monkeypatch) -> None:
    """A failed atomic write raises :class:`StorageError`."""
    p = tmp_path / "r.json"
    reg = JsonDocumentRegistry(p)
    monkeypatch.setattr(
        "raghub.storage.json_registry.atomic_write_json",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(StorageError):
        reg.save()
