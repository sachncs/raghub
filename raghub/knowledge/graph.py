"""GraphRAG entity/relation index and its pure helpers.

Owns the :class:`GraphIndex` (entity extraction + community
summaries), the JSON / tokenisation helpers used by extraction
and search, and the asyncio/threading shims that let the
synchronous ingest path call the LLM without blocking the event
loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from hashlib import sha256
from typing import Any

from raghub.embedder import Embedder
from raghub.knowledge.raptor import KnowledgeIndex
from raghub.llm import GenerationRequest, Generator
from raghub.models import Chunk, Hit
from raghub.runtime import capture

MIN_TOKEN_LENGTH = 2

EXTRACT_PROMPT = """Extract entities and relations from the passage.
Reply with JSON only — no prose. Schema:
{{"entities": [{{"name": "<str>", "type": "<str>"}}],
  "triples": [{{"subject": "<str>", "predicate": "<str>", "object": "<str>"}}]}}

Passage:
{passage}
"""

COMMUNITY_PROMPT = """Summarise the following entity / relation cluster
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
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) > MIN_TOKEN_LENGTH}


class GraphIndex(KnowledgeIndex):
    """Entity + community graph (Phase 6.3).

    Attributes:
        name: ``"graphrag"``.

    """

    name = "graphrag"

    def __init__(
        self,
        *,
        llm: Generator | None = None,
        embedder: Embedder | None = None,
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
        self.chunks: dict[str, Chunk] = {}
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
            record = Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                version=chunk.version,
                page=chunk.page,
                source_location=chunk.source_location,
                section=chunk.section,
                company=chunk.company,
                owner=chunk.owner,
                department=chunk.department,
                tenant_id=chunk.tenant_id or "",
                text=chunk.text,
                checksum=sha256(chunk.text.encode("utf-8")).hexdigest(),
                metadata={**chunk.metadata, "vector": vector},
            )
            self.chunks[chunk.id] = record
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

    def search_local(self, query: str, top_k: int = 5) -> list[Hit]:
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

    def search_global(self, query: str, top_k: int = 5) -> list[Hit]:
        """Global search: rank community summaries by query similarity."""
        if not self.community_summaries:
            return []
        query_tokens = tokenise(query)
        scored: list[tuple[int, str, float]] = []
        for idx, summary in self.community_summaries.items():
            overlap = float(len(query_tokens & tokenise(summary)))
            scored.append((idx, summary, overlap))
        scored.sort(key=lambda record: record[2], reverse=True)
        out: list[Hit] = []
        for _rank, (idx, summary, score) in enumerate(scored[: int(top_k)]):
            if score <= 0:
                continue
            chunk_id = f"graphrag-community-{idx}"
            communities = self.communities[idx] if idx < len(self.communities) else set()
            record = Chunk(
                id=chunk_id,
                document_id="graphrag://summary",
                version=1,
                page=1,
                source_location="graphrag://summary",
                section="",
                company="",
                owner="",
                department="",
                tenant_id="",
                text=summary,
                checksum=sha256(summary.encode("utf-8")).hexdigest(),
                metadata={
                    "graphrag_community": True,
                    "entities": sorted(communities),
                },
            )
            out.append(Hit(score=score, chunk=record))
        return out

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        """Run combined local-then-global search and deduplicate."""
        local = self.search_local(query, top_k=top_k)
        global_hits = self.search_global(query, top_k=top_k)
        seen: set[str] = set()
        out: list[Hit] = []
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
            parsed = await self.extract(chunk.text)
            if not parsed:
                continue
            entities = parsed.get("entities", []) or []
            triples = parsed.get("triples", []) or []
            for entity in entities:
                name = (entity.get("name") or "").strip()
                if not name:
                    continue
                self.entity_chunks[name].add(chunk.id)
                self.chunk_entities[chunk.id].add(name)
                self.graph.setdefault(name, set())
            for triple in triples:
                s = (triple.get("subject") or "").strip()
                o = (triple.get("object") or "").strip()
                if not s or not o:
                    continue
                self.graph[s].add(o)
                self.graph[o].add(s)

    async def extract(self, text: str) -> dict[str, Any] | None:
        """Run the extraction prompt on a single chunk."""
        if self.llm is None or not text:
            return None
        raw = await self.llm.async_generate(
            GenerationRequest(
                system_prompt="You extract entities and relations.",
                conversation=[],
                context=[],
                question=EXTRACT_PROMPT.format(passage=text[:3000]),
            )
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
            self.communities = [{entities[v] for v in community} for community in partition]
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
            prompt = COMMUNITY_PROMPT.format(
                entities=", ".join(entities[:30]),
                relations="; ".join(relations[:30]) or "(no relations)",
            )
            raw = await self.llm.async_generate(
                GenerationRequest(
                    system_prompt="You summarise communities.",
                    conversation=[],
                    context=[],
                    question=prompt,
                )
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

    def search_text_fallback(self, query: str, top_k: int) -> list[Hit]:
        """Plain text-overlap fallback when no entity matches."""
        tokens = tokenise(query)
        scored: list[tuple[float, str]] = []
        for chunk_id, record in self.chunks.items():
            score = float(len(tokens & tokenise(record.text)))
            if score > 0:
                scored.append((score, chunk_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits: list[Hit] = []
        for score, cid in scored[: int(top_k)]:
            record = self.chunks[cid]
            hits.append(Hit(score=score, chunk=record))
        return hits

    def rank_by_query_relevance(self, query: str, chunk_ids: set[str]) -> list[str]:
        """Rank ``chunk_ids`` by simple query-token overlap."""
        tokens = tokenise(query)
        scored: list[tuple[str, float]] = []
        for cid in chunk_ids:
            record = self.chunks.get(cid)
            if record is None:
                continue
            score = float(len(tokens & tokenise(record.text)))
            scored.append((cid, score))
        scored.sort(key=lambda record: (-record[1], record[0]))
        return [cid for cid, _score in scored]

    def to_hits(self, chunk_ids: list[str], top_k: int) -> list[Hit]:
        """Build :class:`Hit` objects from chunk ids."""
        out: list[Hit] = []
        for cid in chunk_ids[: int(top_k)]:
            record = self.chunks.get(cid)
            if record is None:
                continue
            score = 1.0
            out.append(Hit(score=score, chunk=record))
        return out


def connected_components(graph_like: GraphIndex) -> list[set[str]]:
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


def run_in_thread(graph_like: GraphIndex, chunks: list[Chunk]) -> None:
    """Run :meth:`GraphIndex.drive_extraction` on a daemon thread and join."""
    import threading

    thread = threading.Thread(
        target=lambda: asyncio.run(graph_like.drive_extraction(chunks)),
        daemon=True,
    )
    thread.start()
    thread.join()
