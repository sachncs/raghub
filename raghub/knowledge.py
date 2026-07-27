"""Knowledge representation layer.

The OKF serialisation, in-memory repository, source manifest, and
the two structured-retrieval indexes (RAPTOR + GraphRAG) all share
the knowledge domain. They live in one file even though the
combined line count is large, because the indexes rely on the
OKF helpers and the manifest for their wiring.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from raghub.embeddings import BaseEmbeddingProvider
from raghub.exceptions import KnowledgeError
from raghub.interfaces.knowledge import KnowledgeRepository
from raghub.llm import BaseLLMProvider
from raghub.models import (
    BlockKind,
    Chunk,
    ChunkRecord,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
    RetrievalHit,
)
from raghub.utils import capture

OKF_SCHEMA_VERSION = "0.1"


# ---------------------------------------------------------------------------
# OKF serialisation
# ---------------------------------------------------------------------------


def to_okf(bundle: KnowledgeBundle) -> dict[str, Any]:
    """Serialise ``bundle`` to a plain-OKF dict.

    Args:
        bundle: The bundle to serialise.

    Returns:
        A JSON-serialisable dict conforming to the OKF schema.
    """
    return {
        "$schema": f"okf/{bundle.schema_version or OKF_SCHEMA_VERSION}",
        "bundle_id": bundle.bundle_id,
        "source_uri": bundle.source_uri,
        "checksum": bundle.checksum,
        "language": bundle.language,
        "mime_type": bundle.mime_type,
        "metadata": bundle.metadata,
        "created_at": bundle.created_at.isoformat(),
        "sections": [
            {
                "section_id": section.section_id,
                "index": section.index,
                "heading": section.heading,
                "page_numbers": section.page_numbers,
                "source_location": section.source_location,
                "blocks": [
                    {
                        "block_id": block.block_id,
                        "kind": block.kind.value,
                        "content": block.content,
                        "metadata": block.metadata,
                    }
                    for block in section.blocks
                ],
            }
            for section in bundle.sections
        ],
    }


def from_okf(payload: dict[str, Any] | str) -> KnowledgeBundle:
    """Parse an OKF payload back into a :class:`KnowledgeBundle`.

    Args:
        payload: A dict produced by :func:`to_okf` or a JSON string
            produced by :func:`dumps`.

    Returns:
        The reconstructed :class:`KnowledgeBundle`.

    Raises:
        KnowledgeError: When the payload is structurally invalid.
    """
    if isinstance(payload, str):
        parsed, _ = capture(json.loads, payload)
        if not isinstance(parsed, dict):
            raise KnowledgeError(f"Invalid OKF JSON: expected dict, got {type(parsed).__name__}")
        payload = parsed
    if not isinstance(payload, dict):
        raise KnowledgeError("OKF payload must be a dict")

    sections: list[DocumentSection] = []
    for raw_section in payload.get("sections", []) or []:
        if not isinstance(raw_section, dict):
            raise KnowledgeError("OKF sections must be dicts")
        blocks: list[DocumentBlock] = []
        for raw_block in raw_section.get("blocks", []) or []:
            if not isinstance(raw_block, dict):
                raise KnowledgeError("OKF blocks must be dicts")
            kind_raw = raw_block.get("kind", "text")
            kind, kind_error = capture(BlockKind, kind_raw)
            if kind_error is not None:
                raise KnowledgeError(
                    f"Unknown OKF block kind: {kind_raw!r}"
                ) from kind_error
            blocks.append(
                DocumentBlock(
                    block_id=raw_block.get("block_id", ""),
                    kind=kind,
                    content=raw_block.get("content", "") or "",
                    metadata=raw_block.get("metadata", {}) or {},
                )
            )
        sections.append(
            DocumentSection(
                section_id=raw_section.get("section_id", ""),
                index=int(raw_section.get("index", 0)),
                heading=raw_section.get("heading", "") or "",
                blocks=blocks,
                page_numbers=list(raw_section.get("page_numbers", []) or []),
                source_location=raw_section.get("source_location", "") or "",
            )
        )

    schema = payload.get("$schema", f"okf/{OKF_SCHEMA_VERSION}")
    version = schema.split("/", 1)[-1] if isinstance(schema, str) else OKF_SCHEMA_VERSION

    return KnowledgeBundle(
        bundle_id=payload.get("bundle_id", ""),
        schema_version=str(version),
        source_uri=payload.get("source_uri", ""),
        checksum=payload.get("checksum", "") or "",
        language=payload.get("language", "") or "",
        mime_type=payload.get("mime_type", "") or "",
        metadata=payload.get("metadata", {}) or {},
        sections=sections,
    )


def dumps(bundle: KnowledgeBundle, *, indent: int | None = 2) -> str:
    """Serialise ``bundle`` as a JSON string.

    Args:
        bundle: The bundle to serialise.
        indent: Optional JSON indent.

    Returns:
        A JSON string.
    """
    return json.dumps(to_okf(bundle), indent=indent, ensure_ascii=False)


def loads(payload: str) -> KnowledgeBundle:
    """Parse ``payload`` as JSON and return a :class:`KnowledgeBundle`.

    Args:
        payload: A JSON string.

    Returns:
        The reconstructed bundle.
    """
    data, error = capture(json.loads, payload)
    if error is not None or not isinstance(data, dict):
        raise KnowledgeError(f"Invalid OKF JSON: {error}")
    return from_okf(data)


# ---------------------------------------------------------------------------
# In-memory knowledge repository
# ---------------------------------------------------------------------------


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Threadsafe-ish :class:`KnowledgeRepository` for tests and dev."""

    def __init__(self) -> None:
        """Initialise the empty in-memory store."""
        self.bundles: dict[str, KnowledgeBundle] = {}
        self.by_source: dict[str, list[str]] = {}

    def save(self, bundle: KnowledgeBundle) -> KnowledgeBundle:
        """Persist ``bundle`` in memory."""
        self.bundles[bundle.bundle_id] = bundle
        self.by_source.setdefault(bundle.source_uri, []).insert(0, bundle.bundle_id)
        return bundle

    def get(self, bundle_id: str) -> KnowledgeBundle | None:
        """Return the bundle with id ``bundle_id`` or ``None``."""
        return self.bundles.get(bundle_id)

    def list_by_source(self, source_uri: str) -> list[KnowledgeBundle]:
        """Return every bundle for ``source_uri`` (newest first)."""
        return [
            self.bundles[bid] for bid in self.by_source.get(source_uri, []) if bid in self.bundles
        ]

    def delete(self, bundle_id: str) -> None:
        """Remove the bundle; missing ids are ignored."""
        bundle = self.bundles.pop(bundle_id, None)
        if bundle is not None:
            ids = self.by_source.get(bundle.source_uri, [])
            if bundle_id in ids:
                ids.remove(bundle_id)


# ---------------------------------------------------------------------------
# Source manifest
# ---------------------------------------------------------------------------


class SourceManifest:
    """Persistent index of source URIs and their checksums."""

    def __init__(self, path: Path | str) -> None:
        """Initialise the manifest at ``path``."""
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        text, text_error = capture(self.path.read_text, encoding="utf-8")
        if text_error is None:
            payload, json_error = capture(json.loads, text or "{}")
            if json_error is None and isinstance(payload, dict):
                self.records = {str(k): v for k, v in payload.items() if isinstance(v, dict)}
            self.records = {}

    def save(self) -> None:
        """Persist the manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.records, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record(self, source_uri: str, *, bundle_id: str, checksum: str) -> None:
        """Record or update a source."""
        self.records[source_uri] = {"bundle_id": bundle_id, "checksum": checksum}

    def remove(self, source_uri: str) -> None:
        """Remove a source from the manifest."""
        self.records.pop(source_uri, None)

    def __contains__(self, source_uri: str) -> bool:
        """Check whether a source URI is tracked in the manifest."""
        return source_uri in self.records

    def __getitem__(self, source_uri: str) -> dict[str, Any]:
        """Retrieve the record for a source URI."""
        return self.records[source_uri]

    def items(self) -> Iterable[tuple[str, dict[str, Any]]]:
        """Yield ``(source_uri, record)`` pairs."""
        return self.records.items()

    def sources(self) -> list[str]:
        """Return the list of known source URIs."""
        return list(self.records.keys())


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Knowledge-index base (Phase 6.1)
# ---------------------------------------------------------------------------


class KnowledgeIndex(ABC):
    """Abstract structured-retrieval overlay."""

    name: str = "knowledge_index"

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Ingest a batch of chunks (called once per ``ingest()``)."""

    @abstractmethod
    def delete_for_document(self, document_id: str) -> int:
        """Remove every artefact belonging to ``document_id``."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Search the index for entries relevant to ``query``."""

    def health(self) -> dict[str, Any]:
        """Return a small JSON-friendly status dict for ``/health``."""
        chunks = getattr(self, "chunks", None) or {}
        return {"name": self.name, "chunks": len(chunks)}


# ---------------------------------------------------------------------------
# RAPTOR (Phase 6.2)
# ---------------------------------------------------------------------------


SUMMARY_PROMPT = (
    "Summarise the following passages into one tight paragraph "
    "(3-5 sentences) that captures the shared topic. Reply with "
    "the paragraph only — no preamble, no heading.\n\n"
    "{passages}"
)


class RaptorIndex(KnowledgeIndex):
    """Recursive summary tree (Phase 6.2).

    Attributes:
        name: ``"raptor"``.
    """

    name = "raptor"

    def __init__(
        self,
        *,
        llm: BaseLLMProvider | None = None,
        embedder: BaseEmbeddingProvider | None = None,
        depth: int = 2,
        cluster_size: int = 5,
        max_summary_chars: int = 1500,
    ) -> None:
        """Initialise the index."""
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if cluster_size < 1:
            raise ValueError("cluster_size must be >= 1")
        self.llm = llm
        self.embedder = embedder
        self.depth = int(depth)
        self.cluster_size = int(cluster_size)
        self.max_summary_chars = int(max_summary_chars)
        self.levels: list[list[ChunkRecord]] = []
        self.lock_token = 0

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Add a batch of chunks to the leaf level."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel")
        leaves: list[ChunkRecord] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            record = chunk_to_record(chunk, vector, level=0)
            leaves.append(record)
        self.levels = [leaves]
        self.lock_token += 1
        self.rebuild_tree()

    def delete_for_document(self, document_id: str) -> int:
        """Drop every entry (leaf or summary) tied to ``document_id``."""
        removed = 0
        for level in self.levels:
            before = len(level)
            level[:] = [
                rec for rec in level if rec.document_id != document_id
            ]
            removed += before - len(level)
        self.lock_token += 1
        return removed

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return hits across every level, sorted by cosine similarity."""
        if not self.levels or self.embedder is None:
            return []
        query_vec, _ = capture(self.embedder.embed_text, query)
        if not isinstance(query_vec, list):
            return []
        hits: list[RetrievalHit] = []
        for level in self.levels:
            if not level:
                continue
            for record in level:
                if not record.metadata.get("vector"):
                    continue
                score = cosine(query_vec, record.metadata["vector"])
                hits.append(
                    RetrievalHit(
                        chunk_id=record.chunk_id,
                        score=score,
                        chunk=record,
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: int(top_k)]

    def health(self) -> dict[str, Any]:
        """Return a status dict for ``/health``."""
        counts = {f"level_{i}": len(level) for i, level in enumerate(self.levels)}
        return {"name": self.name, "levels": counts, "lock_token": self.lock_token}

    def rebuild_tree(self) -> None:
        """Recompute the upper summary levels in place."""
        if self.depth <= 0 or self.llm is None or self.embedder is None:
            return
        if not self.levels or not self.levels[0]:
            return
        current = self.levels[0]
        for level_idx in range(1, self.depth + 1):
            clusters = cluster(current, self.cluster_size)
            if len(clusters) <= 1:
                break
            summaries: list[ChunkRecord] = []
            for group in clusters:
                summary_text = summarise(group, self.llm, self.max_summary_chars)
                if not summary_text:
                    continue
                summary_id = summary_id_for(summary_text)
                vec, _ = capture(self.embedder.embed_text, summary_text)
                if not isinstance(vec, list):
                    vec = []
                summaries.append(
                    ChunkRecord(
                        chunk_id=summary_id,
                        document_id=group[0].document_id,
                        version=1,
                        page=group[0].page,
                        source_location="raptor://summary",
                        section="",
                        company=group[0].company,
                        owner=group[0].owner,
                        department="",
                        text=summary_text,
                        metadata={
                            "vector": vec,
                            "raptor_level": level_idx,
                            "members": [c.chunk_id for c in group],
                        },
                    )
                )
            if not summaries:
                break
            self.levels.append(summaries)
            current = summaries


def cluster(
    items: list[ChunkRecord], cluster_size: int
) -> list[list[ChunkRecord]]:
    """Cluster ``items`` by embedding cosine distance."""
    if len(items) <= cluster_size:
        return [items]
    import numpy as np
    from sklearn.cluster import KMeans

    vectors = [
        np.asarray(item.metadata.get("vector") or [0.0], dtype=float)
        for item in items
    ]
    n_clusters = max(1, min(len(items) // cluster_size, len(items)))
    if n_clusters <= 1:
        return [items]
    matrix = np.vstack(vectors)
    kmeans, fit_error = capture(KMeans, n_clusters=n_clusters, n_init=3, random_state=0)
    if fit_error is None:
        labels, fit_error = capture(kmeans.fit_predict, matrix)
    if fit_error is not None:
        return [
            items[i : i + cluster_size]
            for i in range(0, len(items), cluster_size)
        ]
    groups: dict[int, list[ChunkRecord]] = {}
    for label, item in zip(labels, items, strict=True):
        groups.setdefault(int(label), []).append(item)
    return [group for group in groups.values() if group]


async def summarise_async(
    cluster_items: list[ChunkRecord],
    llm: Any,
    max_chars: int,
) -> str:
    """Async summarise a cluster via the LLM."""
    joined = "\n\n---\n\n".join(c.text for c in cluster_items if c.text)
    if not joined:
        return ""
    joined = joined[:max_chars]
    return cast(
        str,
        await llm.async_generate(
            system_prompt="You summarise passages.",
            conversation=[],
            context=[],
            question=SUMMARY_PROMPT.format(passages=joined),
        ),
    )


def summarise(
    cluster_items: list[ChunkRecord],
    llm: Any,
    max_chars: int,
) -> str:
    """Synchronous summarise: drives the async helper."""
    import asyncio

    async def driver() -> str:
        """Thin async shim that calls :func:`summarise_async`."""
        return await summarise_async(cluster_items, llm, max_chars)

    _, error = capture(asyncio.get_running_loop)
    running = error is None
    if running:
        import threading

        result: list[str] = [""]

        def runner() -> None:
            """Drive ``driver`` on the private thread and capture the result."""
            result[0] = asyncio.run(driver())

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        return result[0].strip()
    return asyncio.run(driver()).strip()


def summary_id_for(text: str) -> str:
    """Deterministic id for a summary chunk."""
    return "raptor-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_to_record(chunk: Chunk, vector: list[float], *, level: int) -> ChunkRecord:
    """Convert a canonical :class:`Chunk` into a :class:`ChunkRecord`."""
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        version=chunk.version,
        page=chunk.page,
        source_location=chunk.source_location,
        section=chunk.section,
        company=chunk.company,
        owner=chunk.owner,
        department=chunk.department,
        text=chunk.text,
        metadata={**chunk.metadata, "vector": vector, "raptor_level": level},
    )


# ---------------------------------------------------------------------------
# GraphRAG (Phase 6.3)
# ---------------------------------------------------------------------------


EXTRACT_PROMPT = """Extract entities and relations from the passage.
Reply with JSON only — no prose. Schema:
{{"entities": [{{"name": "<str>", "type": "<str>"}}],
  "triples": [{{"subject": "<str>", "predicate": "<str>", "object": "<str>"}}]}}

Passage:
{passage}
"""

SUMMARISE_COMMUNITY_PROMPT = """Summarise the following entity / relation cluster
in one short paragraph (2-4 sentences). Reply with the paragraph only.

Entities:
{entities}

Relations:
{relations}
"""


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a string."""
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return None
    parsed, _ = capture(json.loads, candidate[start:end])
    return parsed if isinstance(parsed, dict) else None


def tokenise(text: str) -> set[str]:
    """Lower-case word tokens, dropping words of length ≤ 2."""
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) > 2}


class GraphRagIndex(KnowledgeIndex):
    """Entity + community graph (Phase 6.3).

    Attributes:
        name: ``"graphrag"``.
    """

    name = "graphrag"

    def __init__(
        self,
        *,
        llm: BaseLLMProvider | None = None,
        embedder: BaseEmbeddingProvider | None = None,
        hop_limit: int = 2,
    ) -> None:
        """Initialise the index."""
        if hop_limit < 0:
            raise ValueError("hop_limit must be >= 0")
        self.llm = llm
        self.embedder = embedder
        self.hop_limit = int(hop_limit)
        self.graph: dict[str, set[str]] = defaultdict(set)
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        self.chunk_entities: dict[str, set[str]] = defaultdict(set)
        self.communities: list[set[str]] = []
        self.community_summaries: dict[int, str] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.lock_token = 0

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Add chunks; extract entities + triples; rebuild communities."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel")
        for chunk, vector in zip(chunks, vectors, strict=True):
            record = ChunkRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                version=chunk.version,
                page=chunk.page,
                source_location=chunk.source_location,
                section=chunk.section,
                company=chunk.company,
                owner=chunk.owner,
                department=chunk.department,
                text=chunk.text,
                metadata={**chunk.metadata, "vector": vector},
            )
            self.chunks[chunk.chunk_id] = record
        self.extract_and_link(chunks)
        self.partition_communities()
        self.summarise_communities()
        self.lock_token += 1

    def delete_for_document(self, document_id: str) -> int:
        """Drop every entity / triple tied to chunks of ``document_id``."""
        to_drop: set[str] = set()
        for chunk_id, record in list(self.chunks.items()):
            if record.document_id == document_id:
                del self.chunks[chunk_id]
                to_drop.add(chunk_id)
        removed = len(to_drop)
        if not to_drop:
            return 0
        for entity, chunks in list(self.entity_chunks.items()):
            chunks -= to_drop
            if not chunks:
                self.entity_chunks.pop(entity, None)
                self.graph.pop(entity, None)
                for neighbours in list(self.graph.values()):
                    neighbours.discard(entity)
        for chunk_id in to_drop:
            self.chunk_entities.pop(chunk_id, None)
        self.partition_communities()
        self.summarise_communities()
        self.lock_token += 1
        return removed

    def search_local(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Local search: anchor on entities mentioned in the query."""
        anchors = self.entities_in_query(query)
        if not anchors:
            return self.search_text_fallback(query, top_k)
        reached: set[str] = set()
        frontier: set[str] = set(anchors)
        for _ in range(self.hop_limit + 1):
            reached.update(frontier)
            next_frontier: set[str] = set()
            for entity in frontier:
                next_frontier.update(self.graph.get(entity, ()))
            next_frontier -= reached
            if not next_frontier:
                break
            frontier = next_frontier
        chunk_ids: set[str] = set()
        for entity in reached:
            chunk_ids.update(self.entity_chunks.get(entity, ()))
        ranked = self.rank_by_query_relevance(query, chunk_ids)
        return self.to_hits(ranked, top_k)

    def search_global(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Global search: rank community summaries by query similarity."""
        if not self.community_summaries:
            return []
        query_tokens = tokenise(query)
        scored: list[tuple[int, str, float]] = []
        for idx, summary in self.community_summaries.items():
            overlap = float(len(query_tokens & tokenise(summary)))
            scored.append((idx, summary, overlap))
        scored.sort(key=lambda item: item[2], reverse=True)
        out: list[RetrievalHit] = []
        for _rank, (idx, summary, score) in enumerate(scored[: int(top_k)]):
            if score <= 0:
                continue
            chunk_id = f"graphrag-community-{idx}"
            communities = self.communities[idx] if idx < len(self.communities) else set()
            record = ChunkRecord(
                chunk_id=chunk_id,
                document_id="graphrag://summary",
                version=1,
                page=1,
                source_location="graphrag://summary",
                section="",
                company="",
                owner="",
                department="",
                text=summary,
                metadata={
                    "graphrag_community": True,
                    "entities": sorted(communities),
                },
            )
            out.append(RetrievalHit(chunk_id=chunk_id, score=score, chunk=record))
        return out

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Combined search — local first, then global, deduped."""
        local = self.search_local(query, top_k=top_k)
        global_hits = self.search_global(query, top_k=top_k)
        seen: set[str] = set()
        out: list[RetrievalHit] = []
        for hit in (*local, *global_hits):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            out.append(hit)
        return out[: int(top_k)]

    def extract_and_link(self, chunks: list[Chunk]) -> None:
        """Run the LLM extraction and link chunks to entities."""
        if self.llm is None:
            return
        if running_loop_present():
            run_in_thread(self, chunks)
        else:
            asyncio.run(self.drive_extraction(chunks))

    @staticmethod
    def running_loop_present() -> bool:
        """Return ``True`` when an asyncio loop is running in this thread."""
        _, error = capture(asyncio.get_running_loop)
        return error is None

    def run_in_thread(self, chunks: list[Chunk]) -> None:
        """Execute :meth:`drive_extraction` on a daemon thread and join."""
        import threading

        thread = threading.Thread(
            target=lambda: asyncio.run(self.drive_extraction(chunks)),
            daemon=True,
        )
        thread.start()
        thread.join()

    async def drive_extraction(self, chunks: list[Chunk]) -> None:
        """Async body of :meth:`extract_and_link`."""
        for chunk in chunks:
            parsed = await self.extract_async(chunk.text)
            if not parsed:
                continue
            entities = parsed.get("entities", []) or []
            triples = parsed.get("triples", []) or []
            for entity in entities:
                name = (entity.get("name") or "").strip()
                if not name:
                    continue
                self.entity_chunks[name].add(chunk.chunk_id)
                self.chunk_entities[chunk.chunk_id].add(name)
                self.graph.setdefault(name, set())
            for triple in triples:
                s = (triple.get("subject") or "").strip()
                o = (triple.get("object") or "").strip()
                if not s or not o:
                    continue
                self.graph[s].add(o)
                self.graph[o].add(s)

    async def extract_async(self, text: str) -> dict[str, Any] | None:
        """Run the extraction prompt on a single chunk."""
        if self.llm is None or not text:
            return None
        raw = await self.llm.async_generate(
            system_prompt="You extract entities and relations.",
            conversation=[],
            context=[],
            question=EXTRACT_PROMPT.format(passage=text[:3000]),
        )
        return extract_json_object(raw or "")

    def partition_communities(self) -> None:
        """Group entities into communities."""
        import igraph as ig
        import leidenalg as la

        entities = list(self.graph.keys())
        self.communities = []
        if not entities:
            return
        edges: list[tuple[int, int]] = []
        index = {entity: i for i, entity in enumerate(entities)}
        for source, targets in self.graph.items():
            for target in targets:
                s = index.get(source)
                t = index.get(target)
                if s is None or t is None or s == t:
                    continue
                edges.append((s, t))
        if not edges:
            self.communities = [{e} for e in entities]
            return
        g, graph_error = capture(ig.Graph, n=len(entities), edges=edges, directed=False)
        partition, partition_error = (
            capture(la.find_partition, g, la.ModularityVertexPartition)
            if graph_error is None
            else (None, graph_error)
        )
        if graph_error is None and partition_error is None and partition is not None:
            self.communities = [
                {entities[v] for v in community} for community in partition
            ]
        else:
            self.communities = list(connected_components(self))

    def summarise_communities(self) -> None:
        """Render a short summary per community via the LLM."""
        self.community_summaries = {}
        if self.llm is None or not self.communities:
            return
        if running_loop_present():
            import threading

            thread = threading.Thread(
                target=lambda: asyncio.run(self.drive_summarisation()),
                daemon=True,
            )
            thread.start()
            thread.join()
        else:
            asyncio.run(self.drive_summarisation())

    async def drive_summarisation(self) -> None:
        """Async body of :meth:`summarise_communities`."""
        if self.llm is None:
            return
        for idx, community in enumerate(self.communities):
            entities = sorted(community)
            relations: list[str] = []
            for source in community:
                for target in self.graph[source]:
                    if target in community:
                        relations.append(f"{source} ↔ {target}")
            prompt = SUMMARISE_COMMUNITY_PROMPT.format(
                entities=", ".join(entities[:30]),
                relations="; ".join(relations[:30]) or "(no relations)",
            )
            raw = await self.llm.async_generate(
                system_prompt="You summarise communities.",
                conversation=[],
                context=[],
                question=prompt,
            )
            summary = (raw or "").strip() or f"Community of {len(community)} entities."
            self.community_summaries[idx] = summary[:600]

    def entities_in_query(self, query: str) -> set[str]:
        """Find entities mentioned in the query."""
        lowered = query.lower()
        out: set[str] = set()
        for entity in self.graph:
            if entity.lower() in lowered:
                out.add(entity)
        return out

    def search_text_fallback(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Plain text-overlap fallback when no entity matches."""
        tokens = tokenise(query)
        scored: list[tuple[float, str]] = []
        for chunk_id, record in self.chunks.items():
            score = float(len(tokens & tokenise(record.text)))
            if score > 0:
                scored.append((score, chunk_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, cid in scored[: int(top_k)]:
            record = self.chunks[cid]
            hits.append(RetrievalHit(chunk_id=cid, score=score, chunk=record))
        return hits

    def rank_by_query_relevance(
        self, query: str, chunk_ids: set[str]
    ) -> list[str]:
        """Rank ``chunk_ids`` by simple query-token overlap."""
        tokens = tokenise(query)
        scored: list[tuple[str, float]] = []
        for cid in chunk_ids:
            record = self.chunks.get(cid)
            if record is None:
                continue
            score = float(len(tokens & tokenise(record.text)))
            scored.append((cid, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [cid for cid, _score in scored]

    def to_hits(
        self, chunk_ids: list[str], top_k: int
    ) -> list[RetrievalHit]:
        """Build :class:`RetrievalHit` objects from chunk ids."""
        out: list[RetrievalHit] = []
        for cid in chunk_ids[: int(top_k)]:
            record = self.chunks.get(cid)
            if record is None:
                continue
            score = 1.0
            out.append(RetrievalHit(chunk_id=cid, score=score, chunk=record))
        return out


def connected_components(graph_like: GraphRagIndex) -> list[set[str]]:
    """Networkx-free connected components over the graph field."""
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in graph_like.graph:
        if node in visited:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(graph_like.graph[current] - visited)
        components.append(component)
    return components


def running_loop_present() -> bool:
    """Return ``True`` when an asyncio loop is running in this thread."""
    _, error = capture(asyncio.get_running_loop)
    return error is None


def run_in_thread(graph_like: GraphRagIndex, chunks: list[Chunk]) -> None:
    """Run :meth:`GraphRagIndex.drive_extraction` on a daemon thread and join."""
    import threading

    thread = threading.Thread(
        target=lambda: asyncio.run(graph_like.drive_extraction(chunks)),
        daemon=True,
    )
    thread.start()
    thread.join()