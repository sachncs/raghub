"""Concurrency tests for InMemoryVectorStore.

Verifies thread-safety under concurrent insert, search, and delete
operations using the store's built-in RLock.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from raghub.models import ChunkRecord, Classification
from raghub.vectorstore import InMemoryVectorStore


def _make_chunk(chunk_id: str, company: str = "acme", document_id: str = "d1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        version=1,
        text=f"text for {chunk_id}",
        company=company,
        owner="test@test.com",
        classification=Classification.INTERNAL,
    )


class TestConcurrentInsertAndSearch:
    def test_concurrent_inserts_dont_corrupt_store(self) -> None:
        store = InMemoryVectorStore()

        def insert_chunk(i: int) -> None:
            chunk = _make_chunk(f"c{i}")
            store.insert([chunk], [[float(i)] * 4])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(insert_chunk, i) for i in range(100)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()  # propagate exceptions

        assert len(store.records) == 100

    def test_concurrent_inserts_and_searches(self) -> None:
        store = InMemoryVectorStore()

        # Pre-populate some data
        for i in range(10):
            chunk = _make_chunk(f"seed{i}")
            store.insert([chunk], [[1.0, 0.0, 0.0, 0.0]])

        errors: list[Exception] = []

        def insert_chunk(i: int) -> None:
            try:
                chunk = _make_chunk(f"c{i}")
                store.insert([chunk], [[0.0, 1.0, 0.0, 0.0]])
            except Exception as e:
                errors.append(e)

        def search_store() -> None:
            try:
                results = store.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
                # Results must always be a list (never corrupted)
                assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i in range(50):
                futures.append(pool.submit(insert_chunk, i))
                futures.append(pool.submit(search_store))
            concurrent.futures.wait(futures)

        assert not errors, f"Concurrent operations raised: {errors}"

    def test_concurrent_delete_and_search(self) -> None:
        store = InMemoryVectorStore()

        # Pre-populate
        for i in range(20):
            chunk = _make_chunk(f"c{i}")
            store.insert([chunk], [[float(i)] * 4])

        errors: list[Exception] = []

        def delete_chunk(chunk_id: str) -> None:
            try:
                store.delete([chunk_id])
            except Exception as e:
                errors.append(e)

        def search_store() -> None:
            try:
                results = store.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
                assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i in range(0, 20, 2):
                futures.append(pool.submit(delete_chunk, f"c{i}"))
                futures.append(pool.submit(search_store))
            concurrent.futures.wait(futures)

        assert not errors, f"Concurrent operations raised: {errors}"
        # Half the records should be deleted
        assert len(store.records) == 10

    def test_concurrent_delete_document_and_insert(self) -> None:
        store = InMemoryVectorStore()

        # Pre-populate
        for i in range(5):
            chunk = _make_chunk(f"c{i}", document_id="doc-to-delete")
            store.insert([chunk], [[1.0, 0.0, 0.0, 0.0]])

        for i in range(5):
            chunk = _make_chunk(f"other{i}", document_id="doc-to-keep")
            store.insert([chunk], [[0.0, 1.0, 0.0, 0.0]])

        errors: list[Exception] = []

        def delete_document() -> None:
            try:
                store.delete_document("doc-to-delete")
            except Exception as e:
                errors.append(e)

        def insert_new() -> None:
            try:
                chunk = _make_chunk(f"new{i}", document_id="doc-new")
                store.insert([chunk], [[0.0, 0.0, 1.0, 0.0]])
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(delete_document),
                pool.submit(insert_new),
            ]
            concurrent.futures.wait(futures)

        assert not errors, f"Concurrent operations raised: {errors}"
        # "doc-to-delete" records should be gone
        remaining = [r for r in store.records.values() if r.chunk.document_id == "doc-to-delete"]
        assert len(remaining) == 0
