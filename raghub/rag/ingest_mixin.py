"""Ingest mixin for the RAG facade.

Holds the ingestion entry points (:meth:`ingest`, :meth:`aingest`,
:meth:`ingest_directory`, :meth:`ingest_dir`, :meth:`settings_path`,
:meth:`ingest_one`, :meth:`delete`) and the background-job helpers
(:meth:`ingest_async`, :meth:`job_status`).

The mixin assumes the host class has already wired the
collaborators it needs:

- ``self.ingest_pipeline`` :class:`Ingest` instance
- ``self.manifest`` :class:`Manifest` for incremental indexing
- ``self.vector_store`` and ``self.knowledge_repo`` for deletion
- ``self.queue_`` (or ``self.background_ingestion``) for async jobs
- ``self.settings`` and ``self.embedder`` for the per-file worker
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from raghub.errors import IngestionError
from raghub.ingest import Resumable
from raghub.knowledge import sha256_bytes
from raghub.models import Pipeline, PipelineCtx, deterministic_id
from raghub.rag.defaults import ingest_one_worker
from raghub.types import JSONValue


class IngestMixin:
    """Mixin providing synchronous, async, and background ingestion."""

    def ingest(
        self,
        source: str | Path | bytes,
        **options: JSONValue,
    ) -> Pipeline:
        """Ingest a file, directory, or raw bytes synchronously.

        Args:
            source: Path to a file/directory or raw bytes.
            **options: Optional overrides (``source_uri=``,
                ``mime_type=``, ``metadata=``, ``force=``,
                ``user=``).

        Returns:
            A :class:`Pipeline` for a single source, or a
            composite result for a directory.

        Raises:
            IngestionError: When ingestion cannot complete.

        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                files = sorted(p for p in p.rglob("*") if p.is_file())
                results: list[Pipeline] = []
                iterator = tqdm(files, desc="Ingesting", unit="file")
                for child in iterator:
                    results.append(self.ingest(child, metadata=options.get("metadata"), user=options.get("user")))
                return Pipeline(
                    pipeline_id="batch",
                    pipeline_name="ingest",
                    outputs={"batch": results},
                )
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = options.get("source_uri") or "bytes://memory"
        if not file_bytes:
            raise IngestionError(f"ingest({source!r}) received empty bytes; nothing to index.")
        from raghub.coroutines import maybe_run as maybe_await

        result = cast(
            Pipeline,
            maybe_await(
                self.ingest_one(
                    file_bytes,
                    uri,
                    options.get("mime_type", "text/plain"),
                    metadata=options.get("metadata"),
                    force=options.get("force", False),
                    user=options.get("user"),
                )
            ),
        )
        if getattr(result, "error", None) is not None:
            raise IngestionError(
                f"ingest({source!r}) failed: {result.error.message if result.error else 'unknown'}"
            )
        return result

    async def aingest(
        self,
        source: str | Path | bytes,
        **options: JSONValue,
    ) -> Pipeline:
        """Async version of :meth:`ingest`.

        Args:
            source: Path to a file/directory or raw bytes.
            **options: Optional overrides (``source_uri=``,
                ``mime_type=``, ``metadata=``, ``force=``,
                ``user=``).

        Raises:
            IngestionError: When ingestion cannot complete.

        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                return await self.ingest_directory(
                    p, options.get("metadata"), options.get("user")
                )
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = options.get("source_uri") or "bytes://memory"
        if not file_bytes:
            raise IngestionError(f"aingest({source!r}) received empty bytes; nothing to index.")
        return await self.ingest_one(
            file_bytes,
            uri,
            options.get("mime_type", "text/plain"),
            metadata=options.get("metadata"),
            force=options.get("force", False),
            user=options.get("user"),
        )

    async def ingest_directory(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
    ) -> Pipeline:
        """Recursively ingest a directory asynchronously.

        Args:
            directory: Directory to walk.
            metadata: Optional per-file metadata.
            user: Optional :class:`User`.
            show_progress: When ``True`` (default), wrap the file loop
                in a tqdm progress bar. Suppress with ``False`` for
                non-interactive callers.

        """
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        n_workers = max(1, min(4, len(files)))
        semaphore = asyncio.Semaphore(n_workers)

        async def bounded(child: Path) -> Pipeline:
            """Run ingest on ``child`` under the concurrency cap."""
            async with semaphore:
                return await self.aingest(child, metadata=metadata, user=user)

        results = await asyncio.gather(*(bounded(c) for c in files))
        vector_store = getattr(self, "vector_store", None)
        rebuild = getattr(vector_store, "rebuild_index", None)
        if callable(rebuild):
            rebuild()
        return Pipeline(
            pipeline_id="batch",
            pipeline_name="ingest",
            outputs={"batch": list(results)},
        )

    async def ingest_dir(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
        max_workers: int | None = None,
    ) -> Pipeline:
        """Run every file in ``directory`` through a ProcessPoolExecutor.

        Each worker process builds its own RAG from a serialised
        settings path (cheap — no RAG stack re-initialisation since
        ``RAG.__init__`` only allocates slots). The worker returns the
        list of (Chunk, vector) tuples it would have inserted
        into the local store. The main process inserts them into the
        shared vector store and rebuilds BM25 once at the end.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        files = sorted(p for p in directory.rglob("*") if p.is_file())
        if not files:
            return Pipeline(
                pipeline_id="batch",
                pipeline_name="ingest",
                outputs={"batch": []},
            )

        n_workers = max(1, min(max_workers or os.cpu_count() or 4, len(files)))
        settings_path = self.settings_path()
        embedder_signature = (self.embedder.model_name, self.embedder.dimension)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(
                    ingest_one_worker,
                    settings_path,
                    str(p),
                    metadata,
                    embedder_signature,
                )
                for p in files
            ]
            worker_outputs = [f.result() for f in futures]

        vector_store = getattr(self, "vector_store", None)
        n_inserted = 0
        for chunks, vectors in worker_outputs:
            if chunks and vector_store is not None:
                written = vector_store.insert(chunks, vectors)
                n_inserted += written
        rebuild = getattr(vector_store, "rebuild_index", None)
        if callable(rebuild):
            rebuild()

        return Pipeline(
            pipeline_id="batch",
            pipeline_name="ingest",
            outputs={"batch": worker_outputs, "files": [str(p) for p in files]},
        )

    def settings_path(self) -> str:
        """Write the active settings to a sidecar file and return its path.

        Workers re-build ``Settings`` from the file rather than from
        the live :class:`RAG` instance. We round-trip the existing
        ``Settings`` object so any env-var-driven defaults are picked
        up.
        """
        import json
        import tempfile

        path = Path(tempfile.mkstemp(prefix="rag-settings-", suffix=".json")[1])
        path.write_text(
            json.dumps(
                self.settings.model_dump(mode="json"),
                default=str,
            ),
            encoding="utf-8",
        )
        return str(path)

    async def ingest_one(
        self,
        file_bytes: bytes,
        source_uri: str,
        mime_type: str,
        **options: JSONValue,
    ) -> Pipeline:
        """Run a single ingest pipeline asynchronously.

        Args:
            file_bytes: Raw bytes to ingest.
            source_uri: Stable source URI for the file.
            mime_type: MIME hint for the converter.
            **options: Optional overrides (``metadata=``,
                ``force=``, ``user=``).

        """
        user: Any | None = options.get("user")
        context = PipelineCtx(
            pipeline_name="ingest",
            metadata={"user_id": getattr(user, "email", None)} if user is not None else {},
        )
        result = await self.ingest_pipeline.run(
            context,
            file_bytes=file_bytes,
            source_uri=source_uri,
            mime_type=mime_type,
            metadata=options.get("metadata") or {},
            force=options.get("force", False),
            user=user,
        )
        if getattr(result, "error", None) is not None:
            raise IngestionError(result.error or "ingestion failed")
        if hasattr(self.manifest, "record") and hasattr(self.manifest, "get"):
            prior = self.manifest.get(source_uri)
            if not (isinstance(prior, dict) and prior.get("checksum") == sha256_bytes(file_bytes)):
                bundle_id = deterministic_id("bundle", source_uri, sha256_bytes(file_bytes))
                self.manifest.record(
                    source_uri,
                    bundle_id=bundle_id,
                    checksum=sha256_bytes(file_bytes),
                )
        return result

    def delete(self, document_id: str) -> None:
        """Delete a document and all of its chunks.

        Accepts either a bundle id (the deterministic
        ``document_id`` recorded on each chunk), a source URI (the
        ``source_uri`` argument supplied to :meth:`ingest`), or any
        prior bundle id that has been retired to that source. All
        matching bundles are removed from both the vector store and
        the knowledge repository so a subsequent ingest does not see
        stale entries.
        """
        target_ids: set[str] = {document_id}
        if hasattr(self.knowledge_repo, "list_by_source"):
            for bundle in self.knowledge_repo.list_by_source(document_id):
                target_ids.add(bundle.bundle_id)
        if hasattr(self.manifest, "sources"):
            for prior_uri in list(self.manifest.sources()):
                if prior_uri == document_id:
                    prior_record = self.manifest[prior_uri]
                    prior_bundle_id = str(prior_record.get("bundle_id", ""))
                    if prior_bundle_id:
                        target_ids.add(prior_bundle_id)
        for tid in target_ids:
            if hasattr(self.vector_store, "delete_document"):
                self.vector_store.delete_document(tid)
            if hasattr(self.knowledge_repo, "delete"):
                self.knowledge_repo.delete(tid)
            for index in (getattr(self, "raptor", None), getattr(self, "graph", None)):
                if index is not None and hasattr(index, "delete_for_document"):
                    index.delete_for_document(tid)
        if hasattr(self.manifest, "remove"):
            if document_id in self.manifest:
                self.manifest.remove(document_id)
            sources_method = getattr(self.manifest, "sources", None)
            if callable(sources_method):
                for uri in list(sources_method()):
                    record = self.manifest.get(uri)
                    if not isinstance(record, dict):
                        continue
                    if str(record.get("bundle_id", "")) in target_ids:
                        self.manifest.remove(uri)

    def ingest_async(
        self,
        source: str | Path | bytes,
        *,
        source_uri: str | None = None,
        mime_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        user: Any | None = None,
    ) -> str:
        """Submit an ingest job to the background service.

        Routing:
            * If ``self.queue_`` is a :class:`SqliteQueue`
              (constructed in :meth:`__init__` when
              ``Settings.queue.backend == "sqlite"``), the job is
              submitted to that queue and the queue's UUID-shaped
              job id is returned.
            * Otherwise, falls back to the legacy ``Resumable``
              threadpool path.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = source_uri or "bytes://memory"

        if self.queue_ is not None:
            from raghub.jobs import JobStatus
            from raghub.tenants import current, validate_tenant

            tenant_id: str | None = None
            ctx = current()
            if ctx is not None:
                tenant_id = ctx.tenant_id
                validate_tenant(tenant_id)

            content_hash = hashlib.sha256(file_bytes).hexdigest()
            payload = {
                "source": file_bytes.decode("latin-1"),
                "source_uri": uri,
                "mime_type": mime_type,
                "metadata": metadata or {},
                "user": getattr(user, "user_id", None) if user else None,
                "content_hash": content_hash,
            }

            async def submit() -> str:
                existing_jobs = await self.queue_.list_for_tenant(
                    tenant_id=tenant_id,
                    content_hash=content_hash,
                )
                for job in existing_jobs:
                    if (
                        job.payload.get("content_hash") == content_hash
                        and job.status in (
                            JobStatus.PENDING,
                            JobStatus.RUNNING,
                        )
                    ):
                        return job.id
                return await self.queue_.submit(
                    kind="ingest",
                    payload=payload,
                    tenant_id=tenant_id,
                )

            try:
                asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(lambda: asyncio.run(submit())).result()
            except RuntimeError:
                return asyncio.run(submit())

        if self.background_ingestion is None:
            self.background_ingestion = Resumable(
                db_path=self.settings.data_dir / "ingestion_jobs.db"
            )

        return cast(
            str,
            self.background_ingestion.submit(
                self.ingest,
                source=file_bytes,
                source_uri=uri,
                mime_type=mime_type,
                metadata=metadata,
                user=user,
            ),
        )

    def job_status(self, job_id: str) -> str | None:
        """Return the status of a background ingestion job."""
        if self.queue_ is not None:

            async def lookup() -> str | None:
                stats = await self.queue_.stats()
                if sum(stats.values()) == 0:
                    return None
                jobs = await self.queue_.list(status=None, limit=1000)
                for job in jobs:
                    if job.id == job_id:
                        return str(job.status.value)
                return None

            try:
                asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(lambda: asyncio.run(lookup())).result()
            except RuntimeError:
                return asyncio.run(lookup())

        if self.background_ingestion is None:
            return None
        return cast(str | None, self.background_ingestion.get_status(job_id))
