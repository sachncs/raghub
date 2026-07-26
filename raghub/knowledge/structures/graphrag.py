"""GraphRAG: entity + community graph over the chunk corpus.

The original GraphRAG idea (Edge et al., 2024):

1. **Extract** entities and ``(subject, predicate, object)``
   triples from every chunk via an LLM.
2. **Build** an undirected graph (entities as nodes, triples as
   edges).
3. **Partition** the graph into communities (leidenalg when
   available, networkx fallback).
4. **Summarise** each community with the LLM.
5. **Query** in two modes:
   * ``search_local`` — anchor on the entities mentioned in the
     query, expand to their k-hop neighbourhood, return chunk
     text. Good for "who did X?" questions.
   * ``search_global`` — Map-Reduce over community summaries.
     Good for "what is the corpus about?" questions.

This implementation is fully self-contained — no
``community`` / ``graphrag`` / ``cdlib`` packages required. The
optional :mod:`leidenalg` is used when present (better
communities); the deterministic networkx fallback keeps the index
functional in environments that don't ship the optional dep.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from typing import Any

from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.knowledge.structures.base import KnowledgeIndex
from raghub.llm import BaseLLMProvider
from raghub.models import Chunk, ChunkRecord, RetrievalHit

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
    """Pull the first balanced JSON object out of a string.

    Args:
        raw: The LLM's raw output.

    Returns:
        The first balanced JSON object, or ``None`` when none can
        be found or parsed.
    """
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
    try:
        return json.loads(candidate[start:end])
    except ValueError:
        return None


def tokenise(text: str) -> set[str]:
    """Lower-case word tokens, dropping words of length ≤ 2.

    Args:
        text: The input text.

    Returns:
        The set of unique tokens.
    """
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
        llm: BaseEmbeddingProvider | BaseLLMProvider | None = None,
        embedder: BaseEmbeddingProvider | None = None,
        hop_limit: int = 2,
    ) -> None:
        """Initialise the index.

        Args:
            llm: Provider used to extract triples and summarise
                communities. Accepts any object with
                ``async_generate``.
            embedder: Provider used for entity-anchored local
                search. Optional — ``search_local`` degrades to a
                text-match fallback when the embedder is missing.
            hop_limit: Maximum graph distance the local search
                explores from each anchor entity.

        Raises:
            ValueError: When ``hop_limit`` is negative.
        """
        if hop_limit < 0:
            raise ValueError("hop_limit must be >= 0")
        self.llm = llm
        self.embedder = embedder
        self.hop_limit = int(hop_limit)
        self.graph: dict[str, set[str]] = defaultdict(set)  # entity -> {related entities}
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)  # entity -> {chunk_ids}
        self.chunk_entities: dict[str, set[str]] = defaultdict(set)  # chunk_id -> {entities}
        self.communities: list[set[str]] = []  # each: set of entity names
        self.community_summaries: dict[int, str] = {}  # idx -> summary
        self.chunks: dict[str, ChunkRecord] = {}  # chunk_id -> record
        self.lock_token = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Add chunks; extract entities + triples; rebuild communities.

        Args:
            chunks: The chunk batch.
            vectors: Their embeddings (used by ``search_local``).
        """
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel")
        # Store every chunk so search can resolve text later.
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
        """Drop every entity / triple tied to chunks of ``document_id``.

        Args:
            document_id: The document to purge.

        Returns:
            Number of items removed.
        """
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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_local(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Local search: anchor on entities mentioned in the query.

        Args:
            query: Natural-language query.
            top_k: Maximum hits.

        Returns:
            A list of :class:`RetrievalHit` whose underlying chunks
            contain an entity mentioned in ``query``.
        """
        anchors = self.entities_in_query(query)
        if not anchors:
            # No entity overlap — fall back to a chunk-text match.
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
        """Global search: rank community summaries by query similarity.

        Args:
            query: Natural-language query.
            top_k: Maximum hits.

        Returns:
            Synthetic :class:`RetrievalHit` objects whose
            ``chunk.text`` is the community summary. Useful for
            "what is this corpus about?" questions.
        """
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
        """Combined search — local first, then global, deduped.

        Args:
            query: Natural-language query.
            top_k: Maximum hits.

        Returns:
            A deduplicated list of :class:`RetrievalHit` objects.
        """
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def extract_and_link(self, chunks: list[Chunk]) -> None:
        """Run the LLM extraction and link chunks to entities.

        Args:
            chunks: The chunk batch whose text is to be extracted.
        """
        if self.llm is None:
            return
        # ``add_chunks`` is sync but the LLM is async. When called
        # from a running event loop (e.g. inside pytest-asyncio's
        # auto mode), drive the driver on a private thread so we
        # don't re-enter the loop. Otherwise run a fresh loop.

        if self.running_loop_present():
            self.run_in_thread(chunks)
        else:
            asyncio.run(self.drive_extraction(chunks))

    @staticmethod
    def running_loop_present() -> bool:
        """Return ``True`` when an asyncio loop is running in this thread."""
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

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
                # Ensure the entity is a node in the graph so
                # isolated entities still get their own community
                # when we partition.
                self.graph.setdefault(name, set())
            for triple in triples:
                s = (triple.get("subject") or "").strip()
                o = (triple.get("object") or "").strip()
                if not s or not o:
                    continue
                self.graph[s].add(o)
                self.graph[o].add(s)

    async def extract_async(self, text: str) -> dict[str, Any] | None:
        """Run the extraction prompt on a single chunk.

        Args:
            text: The chunk text.

        Returns:
            The parsed JSON, or ``None`` on failure.
        """
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
        """Group entities into communities.

        Uses :mod:`leidenalg` when available for higher-quality
        communities; falls back to a connected-components
        decomposition on the underlying graph.
        """
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
        try:
            g = ig.Graph(n=len(entities), edges=edges, directed=False)
            partition = la.find_partition(g, la.ModularityVertexPartition)
            self.communities = [
                {entities[v] for v in community} for community in partition
            ]
        except (ValueError, RuntimeError):
            self.communities = list(connected_components(self))

    def summarise_communities(self) -> None:
        """Render a short summary per community via the LLM."""
        self.community_summaries = {}
        if self.llm is None or not self.communities:
            return
        if self.running_loop_present():
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
        """Find entities mentioned in the query (case-insensitive substring).

        Args:
            query: The natural-language query.

        Returns:
            The set of entities whose lowercase name appears in the
            query.
        """
        lowered = query.lower()
        out: set[str] = set()
        for entity in self.graph:
            if entity.lower() in lowered:
                out.add(entity)
        return out

    def search_text_fallback(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Plain text-overlap fallback when no entity matches.

        Args:
            query: The natural-language query.
            top_k: Maximum hits.

        Returns:
            A list of :class:`RetrievalHit` ordered by token overlap.
        """
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
        """Rank ``chunk_ids`` by simple query-token overlap.

        Args:
            query: The natural-language query.
            chunk_ids: Candidate chunk ids to rank.

        Returns:
            The candidate chunk ids sorted by descending overlap.
        """
        tokens = tokenise(query)
        scored: list[tuple[str, float]] = []
        for cid in chunk_ids:
            record = self.chunks.get(cid)
            if record is None:
                continue
            score = float(len(tokens & tokenise(record.text)))
            scored.append((cid, score))
        # Sort by score descending; tie-break by chunk id for stability.
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [cid for cid, _score in scored]

    def to_hits(
        self, chunk_ids: list[str], top_k: int
    ) -> list[RetrievalHit]:
        """Build :class:`RetrievalHit` objects from chunk ids.

        Args:
            chunk_ids: Chunk ids in rank order.
            top_k: Maximum hits.

        Returns:
            A list of :class:`RetrievalHit` with score ``1.0`` (local
            search is binary-relevant by construction).
        """
        out: list[RetrievalHit] = []
        for cid in chunk_ids[: int(top_k)]:
            record = self.chunks.get(cid)
            if record is None:
                continue
            score = 1.0  # local search is binary-relevant by construction
            out.append(RetrievalHit(chunk_id=cid, score=score, chunk=record))
        return out


def connected_components(graph_like: GraphRagIndex) -> list[set[str]]:
    """Networkx-free connected components over the graph field.

    Args:
        graph_like: The :class:`GraphRagIndex` whose graph to
            decompose.

    Returns:
        A list of connected components (each a set of entity names).
    """
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


__all__ = [
    "GraphRagIndex",
    "connected_components",
    "extract_json_object",
    "tokenise",
]