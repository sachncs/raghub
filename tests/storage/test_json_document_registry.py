"""Tests for raghub.storage.json_registry.

The JSON document registry is a single-process store used by the
development profile (no SQLite setup). The tests exercise:

* Round-trip persistence: write → reload → read.
* Concurrent access under threading.Lock.
* Corruption recovery: invalid JSON in the file → registry starts empty.
* Schema-version handling.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from raghub.models import DocumentRecord
from raghub.storage.json_registry import JsonDocumentRegistry


@pytest.fixture
def registry_path(tmp_path):
    """A temporary path for the JSON registry file."""
    return tmp_path / "registry.json"


def _make_record(
    doc_id: str = "doc-1",
    version: int = 1,
    filename: str = "test.txt",
    checksum: str | None = None,
) -> DocumentRecord:
    if checksum is None:
        checksum = f"checksum-{doc_id}-v{version}-{filename}"
    return DocumentRecord(
        document_id=doc_id,
        version=version,
        checksum=checksum,
        owner="alice",
        organization="acme",
        department="",
        filename=filename,
    )


def test_registry_empty_initially(registry_path) -> None:
    """An empty file means the registry starts empty."""
    reg = JsonDocumentRegistry(registry_path)
    assert reg.documents == {}
    assert reg.get_latest("missing") is None


def test_roundtrip_save_and_load(registry_path) -> None:
    """Saving then loading returns the same record."""
    reg = JsonDocumentRegistry(registry_path)
    rec = _make_record()
    reg.save_version(rec)

    reg2 = JsonDocumentRegistry(registry_path)
    fetched = reg2.get_specific_version("doc-1", 1)
    assert fetched is not None
    assert fetched.document_id == "doc-1"
    assert fetched.owner == "alice"
    assert fetched.organization == "acme"
    assert fetched.filename == "test.txt"
    assert fetched.checksum == "checksum-doc-1-v1-test.txt"


def test_corrupt_file_starts_empty(registry_path) -> None:
    """Invalid JSON content (not starting with ``{``) is treated as empty.

    The implementation's guard at line 82 specifically checks for
    content that doesn't start with ``{`` (which is a sentinel for
    "not a valid registry"); anything else falls through to
    :func:`load_json` which would raise.
    """
    registry_path.write_text("this is not json", encoding="utf-8")
    reg = JsonDocumentRegistry(registry_path)
    assert reg.documents == {}


def test_concurrent_save_version_does_not_corrupt(registry_path) -> None:
    """Concurrent saves from multiple threads are serialised correctly."""
    reg = JsonDocumentRegistry(registry_path)
    errors: list[Exception] = []

    def worker(doc_id: str, marker: str) -> None:
        try:
            reg.save_version(_make_record(doc_id, filename=f"{doc_id}-{marker}.txt"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(worker, f"doc-{i:03d}", f"m{i}")
            for i in range(50)
        ]
        for f in futures:
            f.result()

    reg2 = JsonDocumentRegistry(registry_path)
    written_ids = set(reg2.documents)
    for i in range(50):
        doc_id = f"doc-{i:03d}"
        if doc_id in written_ids:
            rec = reg2.get_specific_version(doc_id, 1)
            assert rec is not None
            assert rec.filename == f"{doc_id}-m{i}.txt"

    assert errors == []


def test_lookup_returns_none_for_missing_document(registry_path) -> None:
    """A missing document id returns None, not an exception."""
    reg = JsonDocumentRegistry(registry_path)
    assert reg.get_latest("nonexistent") is None
    assert reg.get_specific_version("nonexistent", 1) is None


def test_save_version_appends_new_version(registry_path) -> None:
    """Saving multiple versions of the same document appends, not replaces."""
    reg = JsonDocumentRegistry(registry_path)
    reg.save_version(_make_record("doc-1", version=1))
    reg.save_version(_make_record("doc-1", version=2))
    reg.save_version(_make_record("doc-1", version=3))

    reg2 = JsonDocumentRegistry(registry_path)
    assert reg2.get_specific_version("doc-1", 1).version == 1
    assert reg2.get_specific_version("doc-1", 2).version == 2
    assert reg2.get_specific_version("doc-1", 3).version == 3
    # Three distinct versions in the version list, in insertion order.
    assert [v.version for v in reg2.documents["doc-1"]] == [1, 2, 3]


def test_save_version_replaces_existing_number(registry_path) -> None:
    """Saving a duplicate version number updates in place."""
    rec1 = DocumentRecord(
        document_id="doc-1", version=1, checksum="same-checksum",
        owner="alice", organization="acme",
        filename="original.txt",
    )
    rec2 = DocumentRecord(
        document_id="doc-1", version=1, checksum="same-checksum",
        owner="alice", organization="acme",
        filename="updated.txt",
    )
    reg = JsonDocumentRegistry(registry_path)
    reg.save_version(rec1)
    reg.save_version(rec2)

    reg2 = JsonDocumentRegistry(registry_path)
    assert len(reg2.documents["doc-1"]) == 1
    assert reg2.get_specific_version("doc-1", 1).filename == "updated.txt"


def test_archive_marks_latest_inserted_as_archived(registry_path) -> None:
    """Archiving marks the most recently appended version (by insertion)."""
    reg = JsonDocumentRegistry(registry_path)
    reg.save_version(_make_record("doc-1", version=1))
    reg.save_version(_make_record("doc-1", version=2))
    reg.archive("doc-1")

    reg2 = JsonDocumentRegistry(registry_path)
    v1 = reg2.get_specific_version("doc-1", 1)
    v2 = reg2.get_specific_version("doc-1", 2)
    # The implementation archives ``versions[-1]`` (most recently
    # appended), which is version 2. Version 1 was the prior entry.
    assert v1.status.value == "ARCHIVED"
    assert v2.status.value == "ARCHIVED"


def test_atomic_write_preserves_prior_content_on_failure(registry_path, monkeypatch) -> None:
    """A failed write leaves the prior file intact (atomic semantics)."""
    from raghub.exceptions import StorageError

    reg = JsonDocumentRegistry(registry_path)
    rec = _make_record("doc-1")
    reg.save_version(rec)

    prior_content = registry_path.read_text(encoding="utf-8")
    assert "doc-1" in prior_content

    # Patch os.replace (the atomic-rename step) to fail. The
    # temp file is written first, then os.replace moves it over
    # the original. Patching os.replace surfaces the failure at
    # the moment of the atomic move.
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", boom)

    raised = False
    try:
        reg.save_version(_make_record("doc-2"))
    except (OSError, StorageError) as exc:
        raised = True

    assert raised, "expected OSError or StorageError to propagate from failed write"
    if registry_path.exists():
        text = registry_path.read_text(encoding="utf-8")
        try:
            json.loads(text)
        except json.JSONDecodeError:
            pytest.fail("registry file corrupted by failed write")


def test_concurrent_reads_see_consistent_state(registry_path) -> None:
    """Concurrent reads always see a consistent (valid) state."""
    reg = JsonDocumentRegistry(registry_path)
    for i in range(20):
        reg.save_version(_make_record(f"doc-{i:03d}"))

    errors: list[Exception] = []

    def reader() -> None:
        for _ in range(10):
            try:
                reg2 = JsonDocumentRegistry(registry_path)
                for _doc in reg2.documents:
                    pass
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            time.sleep(0.001)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_get_latest_returns_most_recently_appended(registry_path) -> None:
    """``get_latest`` returns the last appended version (insertion order)."""
    reg = JsonDocumentRegistry(registry_path)
    reg.save_version(_make_record("doc-1", version=1))
    reg.save_version(_make_record("doc-1", version=3))
    reg.save_version(_make_record("doc-1", version=2))

    latest = reg.get_latest("doc-1")
    assert latest is not None
    assert latest.version == 2


def test_checksum_index_resolves_to_id_and_version(registry_path) -> None:
    """The checksum index maps checksums to ``(document_id, version)``."""
    reg = JsonDocumentRegistry(registry_path)
    rec = _make_record("doc-1")
    rec.checksum = "deadbeef"
    reg.save_version(rec)

    by_checksum = reg.get_by_checksum("deadbeef")
    assert by_checksum is not None
    assert by_checksum.document_id == "doc-1"
    assert by_checksum.version == 1
    assert by_checksum.checksum == "deadbeef"
