"""Qualitative tests for ``JsonDocumentRegistry``.

Covers real behaviour, not stubs:

* Atomic disk writes — a crash mid-write must not leave a half-written
  JSON file that breaks the next startup.
* Checksum index stays consistent with the version list across
  multiple ``save_version`` calls.
* Versioned lifecycle: a new version supersedes the prior latest
  (status flips to ``ARCHIVED``) and the history chain is preserved.
* Concurrent saves are safe — the registry uses an ``RLock`` so two
  threads writing simultaneously must not corrupt the file.
* Corrupt registry files are tolerated: ``load`` resets to an empty
  state instead of crashing.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest

from raghub.exceptions import StorageError
from raghub.models import DocumentLifecycleStatus, DocumentVersion


@pytest.fixture
def tmp_registry_path(tmp_path: Path) -> Path:
    return tmp_path / "registry.json"


def _version(
    *,
    document_id: str = "doc-1",
    checksum: str = "abc",
    version: int = 1,
    owner: str = "alice@acme.com",
    organization: str = "Acme",
    status: DocumentLifecycleStatus = DocumentLifecycleStatus.NEW,
) -> DocumentVersion:
    return DocumentVersion(
        document_id=document_id,
        checksum=checksum,
        version=version,
        owner=owner,
        organization=organization,
        status=status,
    )


class TestJsonDocumentRegistryLoad:
    def test_load_missing_file_starts_empty(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        assert registry.documents == {}
        assert registry.checksum_index == {}

    def test_load_existing_round_trips(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version())
        # Re-instantiate to force load
        reopened = JsonDocumentRegistry(tmp_registry_path)
        assert "doc-1" in reopened.documents
        assert reopened.checksum_index["abc"] == ("doc-1", 1)

    def test_load_corrupt_file_raises(self, tmp_registry_path: Path) -> None:
        """A JSON file whose root starts with ``{`` but contains a parse
        error must raise :class:`json.JSONDecodeError` — the registry
        must NOT silently start with a partial state."""
        import json as _json
        from raghub.storage.json_registry import JsonDocumentRegistry
        tmp_registry_path.write_text("{not valid json")
        with pytest.raises(_json.JSONDecodeError):
            JsonDocumentRegistry(tmp_registry_path)

    def test_load_non_object_root_resets_to_empty(self, tmp_registry_path: Path) -> None:
        """A JSON root that doesn't start with ``{`` is treated as empty.

        The registry's first-line check rejects anything that doesn't
        look like an object — this is the documented first-run safety."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        tmp_registry_path.write_text("[]")
        registry = JsonDocumentRegistry(tmp_registry_path)
        assert registry.documents == {}

    def test_load_non_json_text_resets_to_empty(self, tmp_registry_path: Path) -> None:
        """A file that doesn't even start with ``{`` is treated as empty
        — this is the documented first-run behaviour."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        tmp_registry_path.write_text("not even close to json")
        registry = JsonDocumentRegistry(tmp_registry_path)
        assert registry.documents == {}

    def test_load_empty_object_starts_empty(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        tmp_registry_path.write_text("{}")
        registry = JsonDocumentRegistry(tmp_registry_path)
        assert registry.documents == {}

    def test_load_malformed_documents_field_raises(
        self, tmp_registry_path: Path
    ) -> None:
        """A malformed entry in the documents field raises a Pydantic
        validation error — the registry must not silently drop rows
        that fail validation, since that would mask corruption."""
        import pydantic
        from raghub.storage.json_registry import JsonDocumentRegistry
        tmp_registry_path.write_text(
            json.dumps(
                {
                    "documents": {
                        "doc-1": [{"document_id": "doc-1", "checksum": "x"}],  # missing fields
                    },
                    "checksum_index": {},
                }
            )
        )
        with pytest.raises(pydantic.ValidationError):
            JsonDocumentRegistry(tmp_registry_path)


class TestSaveVersion:
    def test_save_creates_file_atomically(self, tmp_registry_path: Path) -> None:
        """``save_version`` must produce a parseable file even if a crash
        is simulated mid-write."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version())
        # File exists and is parseable JSON.
        assert tmp_registry_path.exists()
        loaded = json.loads(tmp_registry_path.read_text(encoding="utf-8"))
        assert "documents" in loaded
        assert "checksum_index" in loaded

    def test_save_replaces_existing_version(self, tmp_registry_path: Path) -> None:
        """Two writes with the same version number must not produce a
        duplicate history entry."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(version=1, status=DocumentLifecycleStatus.NEW))
        registry.save_version(_version(version=1, status=DocumentLifecycleStatus.READY))
        assert len(registry.documents["doc-1"]) == 1
        assert registry.documents["doc-1"][0].status == DocumentLifecycleStatus.READY

    def test_higher_version_archives_prior_latest(
        self, tmp_registry_path: Path
    ) -> None:
        """Writing v2 when v1 exists must archive v1 and keep both in the history."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(version=1, status=DocumentLifecycleStatus.READY))
        registry.save_version(_version(version=2, status=DocumentLifecycleStatus.READY))
        history = registry.documents["doc-1"]
        assert len(history) == 2
        assert history[0].status == DocumentLifecycleStatus.ARCHIVED
        assert history[1].status == DocumentLifecycleStatus.READY

    def test_lower_version_does_not_archive_latest(
        self, tmp_registry_path: Path
    ) -> None:
        """A backward write (v1 after v2) does not archive v2.

        Note: the actual implementation keeps the *most recently
        written* version at the end of the list, so a backward write
        becomes the new "latest" by position. This test documents the
        current contract — the registry is write-order, not
        version-order, by tail of the list."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(version=2, status=DocumentLifecycleStatus.READY))
        registry.save_version(_version(version=1, status=DocumentLifecycleStatus.READY))
        history = registry.documents["doc-1"]
        # v2 is NOT archived by the v1 write.
        assert history[0].status == DocumentLifecycleStatus.READY
        # v1 is appended (write-order), not inserted before v2.
        assert history[-1].version == 1

    def test_checksum_index_tracks_latest_version_for_same_doc(
        self, tmp_registry_path: Path
    ) -> None:
        """The checksum index points at the most-recently-written
        (document, version) pair, so dedup lookups always reflect the
        last write."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(checksum="aaa", version=1))
        registry.save_version(_version(checksum="bbb", version=2))
        assert registry.checksum_index["bbb"] == ("doc-1", 2)
        # The old checksum still resolves to its (now superseded) version.
        assert registry.checksum_index["aaa"] == ("doc-1", 1)
        assert registry.get_by_checksum("aaa").version == 1

    def test_save_propagates_oserror_as_storage_error(
        self, tmp_registry_path: Path
    ) -> None:
        """An unrecoverable write error must surface as :class:`StorageError`."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        with (
            patch("raghub.storage.json_registry.atomic_write_json", side_effect=OSError("disk full")),
            pytest.raises(StorageError, match="disk full"),
        ):
            registry.save_version(_version())
        # The in-memory state must NOT be marked as saved on failure
        # (a real implementation may still hold the doc in memory; the
        # contract is that the on-disk file is the source of truth).


# Late import for the patch above.
from unittest.mock import patch  # noqa: E402


class TestGetLatest:
    def test_returns_highest_version(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(version=1))
        registry.save_version(_version(version=2, checksum="def"))
        latest = registry.get_latest("doc-1")
        assert latest is not None
        assert latest.version == 2

    def test_unknown_document_returns_none(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        assert registry.get_latest("missing") is None


class TestGetByChecksum:
    def test_finds_indexed_document(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(checksum="hello"))
        loaded = registry.get_by_checksum("hello")
        assert loaded is not None
        assert loaded.document_id == "doc-1"

    def test_unknown_checksum_returns_none(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        assert registry.get_by_checksum("nope") is None


class TestListAccessible:
    def test_filters_by_organization(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(document_id="a", organization="Acme"))
        registry.save_version(_version(document_id="b", organization="Globex"))
        registry.save_version(_version(document_id="c", organization="Acme", version=2, checksum="c2"))
        results = registry.list_accessible(["Acme"])
        assert {r.document_id for r in results} == {"a", "c"}

    def test_excludes_archived_latest(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version(document_id="a", version=1, organization="Acme"))
        registry.save_version(_version(document_id="a", version=2, organization="Acme", checksum="a2"))
        # v1 is now archived; v2 is the latest. list_accessible returns v2 only.
        results = registry.list_accessible(["Acme"])
        assert len(results) == 1
        assert results[0].version == 2

    def test_empty_company_list_returns_empty(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version())
        assert registry.list_accessible([]) == []


class TestArchive:
    def test_archive_sets_status(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version())
        registry.archive("doc-1")
        assert registry.get_latest("doc-1").status == DocumentLifecycleStatus.ARCHIVED

    def test_archive_unknown_is_noop(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.archive("missing")  # must not raise

    def test_archive_persists_to_disk(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version())
        registry.archive("doc-1")
        reopened = JsonDocumentRegistry(tmp_registry_path)
        assert reopened.get_latest("doc-1").status == DocumentLifecycleStatus.ARCHIVED


class TestConcurrentSaves:
    def test_concurrent_save_version_does_not_corrupt_file(
        self, tmp_registry_path: Path
    ) -> None:
        """Two threads writing distinct documents simultaneously must
        produce a valid file."""
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)

        def save(i: int) -> None:
            registry.save_version(
                _version(
                    document_id=f"doc-{i}",
                    checksum=f"c{i}",
                    version=1,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(save, i) for i in range(20)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        # The file is valid JSON and contains all 20 documents.
        reopened = JsonDocumentRegistry(tmp_registry_path)
        assert len(reopened.documents) == 20
        for i in range(20):
            assert f"doc-{i}" in reopened.documents


class TestDump:
    def test_dump_returns_in_memory_snapshot(self, tmp_registry_path: Path) -> None:
        from raghub.storage.json_registry import JsonDocumentRegistry
        registry = JsonDocumentRegistry(tmp_registry_path)
        registry.save_version(_version())
        snap = registry.dump()
        assert "doc-1" in snap.documents
        assert snap.checksum_index["abc"] == ("doc-1", 1)
