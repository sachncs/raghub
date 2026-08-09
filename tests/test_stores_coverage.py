"""Coverage tests for :mod:`raghub.stores` (Documents, ImageStore, etc.)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raghub.models import (
    Classification,
    Document,
    DocumentLifecycleStatus,
)
from raghub.stores import (
    Documents,
    ImageStore,
    Snapshot,
)


def _make_document(**overrides: Any) -> Document:
    """Build a Document fixture."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    defaults: dict[str, Any] = {
        "id": "d1",
        "version": 1,
        "checksum": "c1",
        "created_at": now,
        "updated_at": now,
        "owner": "alice@example.com",
        "organization": "acme",
        "classification": Classification.Internal,
    }
    defaults.update(overrides)
    return Document(**defaults)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def test_documents_load_empty_file(tmp_path: Path) -> None:
    """A missing JSON file loads as an empty registry."""
    store = Documents(tmp_path / "registry.json")
    assert store.documents == {}
    assert store.checksum_index == {}


def test_documents_load_malformed_resets_state(tmp_path: Path) -> None:
    """A non-JSON root resets the registry to empty."""
    path = tmp_path / "registry.json"
    path.write_text("not json")
    store = Documents(path)
    assert store.documents == {}


def test_documents_load_persisted_state(tmp_path: Path) -> None:
    """A previously-saved registry is hydrated from disk."""
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "documents": {
                    "d1": [
                        _make_document(id="d1", version=1, checksum="c1").model_dump(mode="json")
                    ]
                },
                "checksum_index": {"c1": ["d1", 1]},
            }
        )
    )
    store = Documents(path)
    assert "d1" in store.documents
    assert store.checksum_index["c1"] == ("d1", 1)


def test_documents_save_writes_to_disk(tmp_path: Path) -> None:
    """``save`` writes the in-memory state to disk."""
    path = tmp_path / "registry.json"
    store = Documents(path)
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    payload = json.loads(path.read_text())
    assert "d1" in payload["documents"]
    assert payload["checksum_index"]["c1"] == ["d1", 1]


def test_documents_save_version_appends_new(tmp_path: Path) -> None:
    """A brand-new version is appended to the version list."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    store.save_version(_make_document(id="d1", version=2, checksum="c2"))
    assert len(store.documents["d1"]) == 2


def test_documents_save_version_replaces_existing(tmp_path: Path) -> None:
    """Writing the same version number replaces the existing record."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    store.save_version(_make_document(id="d1", version=1, checksum="c1-updated"))
    assert len(store.documents["d1"]) == 1
    assert store.documents["d1"][0].checksum == "c1-updated"


def test_documents_save_version_archives_previous(tmp_path: Path) -> None:
    """A new higher version archives the previous latest."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    store.save_version(_make_document(id="d1", version=5, checksum="c5"))
    assert store.documents["d1"][0].status == DocumentLifecycleStatus.Archived


def test_documents_get_latest_returns_highest_version(tmp_path: Path) -> None:
    """``get_latest`` returns the highest-versioned entry."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    store.save_version(_make_document(id="d1", version=3, checksum="c3"))
    doc = store.get_latest("d1")
    assert doc is not None, f"doc should be set by test setup"
    assert doc.version == 3


def test_documents_get_latest_unknown_returns_none(tmp_path: Path) -> None:
    """``get_latest`` returns ``None`` for an unknown id."""
    store = Documents(tmp_path / "registry.json")
    assert store.get_latest("missing") is None


def test_documents_get_version(tmp_path: Path) -> None:
    """``get_version`` returns the requested version."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    store.save_version(_make_document(id="d1", version=2, checksum="c2"))
    doc = store.get_version("d1", 1)
    assert doc is not None, f"doc should be set by test setup"
    assert doc.version == 1


def test_documents_by_checksum(tmp_path: Path) -> None:
    """``by_checksum`` returns the document owning the checksum."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    doc = store.by_checksum("c1")
    assert doc is not None, f"doc should be set by test setup"
    assert doc.id == "d1"


def test_documents_by_checksum_unknown(tmp_path: Path) -> None:
    """``by_checksum`` returns ``None`` for an unknown checksum."""
    store = Documents(tmp_path / "registry.json")
    assert store.by_checksum("missing") is None


def test_documents_list_accessible_filters_by_company(tmp_path: Path) -> None:
    """``list_accessible`` returns docs in the requested company."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", organization="acme"))
    store.save_version(_make_document(id="d2", organization="other"))
    docs = store.list_accessible(["acme"])
    assert {d.id for d in docs} == {"d1"}


def test_documents_list_accessible_skips_archived(tmp_path: Path) -> None:
    """``list_accessible`` excludes archived documents."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", organization="acme", version=1))
    store.save_version(_make_document(id="d1", organization="acme", version=2))
    # First version was archived by the second save_version.
    docs = store.list_accessible(["acme"])
    assert len(docs) == 1
    assert docs[0].version == 2


def test_documents_archive_marks_status(tmp_path: Path) -> None:
    """``archive`` sets the latest version's status to ``ARCHIVED``."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1))
    store.archive("d1")
    doc = store.get_latest("d1")
    assert doc is not None, f"doc should be set by test setup"
    assert doc.status == DocumentLifecycleStatus.Archived


def test_documents_archive_unknown_is_noop(tmp_path: Path) -> None:
    """``archive`` is a no-op for an unknown id."""
    store = Documents(tmp_path / "registry.json")
    store.archive("missing")  # should not raise


def test_documents_dump_returns_snapshot(tmp_path: Path) -> None:
    """``dump`` returns a :class:`Snapshot` of the in-memory state."""
    store = Documents(tmp_path / "registry.json")
    store.save_version(_make_document(id="d1", version=1, checksum="c1"))
    snapshot = store.dump()
    assert isinstance(snapshot, Snapshot)
    assert "d1" in snapshot.documents


# ---------------------------------------------------------------------------
# ImageStore
# ---------------------------------------------------------------------------


def test_image_store_save_writes_file(tmp_path: Path) -> None:
    """``save`` writes the bytes and returns a content hash."""
    store = ImageStore(base_path=tmp_path / "images")
    hash_id = store.save(b"image-bytes", extension=".png")
    assert (tmp_path / "images" / hash_id[:2] / f"{hash_id}.png").exists()


def test_image_store_save_creates_directory(tmp_path: Path) -> None:
    """``save`` creates the base directory if it does not exist."""
    base = tmp_path / "new-dir" / "images"
    store = ImageStore(base_path=base)
    store.save(b"bytes", extension=".jpg")
    assert base.exists()


def test_image_store_get_path_returns_existing(tmp_path: Path) -> None:
    """``get_path`` returns the path when the file exists."""
    store = ImageStore(base_path=tmp_path / "images")
    hash_id = store.save(b"bytes", extension=".png")
    path = store.get_path(hash_id, extension=".png")
    assert path is not None, f"path should be set by test setup"
    assert path.exists()


def test_image_store_get_path_missing_returns_none(tmp_path: Path) -> None:
    """``get_path`` returns ``None`` when the file does not exist."""
    store = ImageStore(base_path=tmp_path / "images")
    assert store.get_path("missing", extension=".png") is None


def test_image_store_get_bytes_returns_existing(tmp_path: Path) -> None:
    """``get_bytes`` returns the bytes when the file exists."""
    store = ImageStore(base_path=tmp_path / "images")
    hash_id = store.save(b"my-image", extension=".png")
    assert store.get_bytes(hash_id, extension=".png") == b"my-image"


def test_image_store_get_bytes_missing_returns_none(tmp_path: Path) -> None:
    """``get_bytes`` returns ``None`` when the file does not exist."""
    store = ImageStore(base_path=tmp_path / "images")
    assert store.get_bytes("missing", extension=".png") is None


def test_image_store_delete_removes_file(tmp_path: Path) -> None:
    """``delete`` removes the file and returns ``True``."""
    store = ImageStore(base_path=tmp_path / "images")
    hash_id = store.save(b"bytes", extension=".png")
    assert store.delete(hash_id, extension=".png") is True
    assert not (tmp_path / "images" / hash_id[:2] / f"{hash_id}.png").exists()


def test_image_store_delete_missing_returns_false(tmp_path: Path) -> None:
    """``delete`` returns ``False`` for an unknown hash."""
    store = ImageStore(base_path=tmp_path / "images")
    assert store.delete("missing", extension=".png") is False
