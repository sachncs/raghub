"""Retrieval support: rerankers, query transformers, fusion, faceted search.

Collapses the previous 18 files (top-level + rerankers/ + transforms/)
into a single helper module. The package surface is small:

Class summary::

    Variant              - one rephrased question (Pydantic value object).
    Transformer          - async rewriter; subclasses implement transform().
    Hyde, MultiQuery, Decompose, StepBack, Compose
                         - concrete transforms; ``Compose`` chains them.
    Rerank               - protocol with rerank() / arerank() variants.
    Identity, Bge, Cohere, Cascade, LlmJudge, Colbert, Context
                         - concrete reranker implementations. ``Context`` is
                           the long-context second pass.
    Fusion               - RRF / linear combine of ranked lists.
    Pipeline             - end-to-end vector + keyword + hybrid retrieval.
    Search               - faceted chunk search with filters.
    RerankerFactory      - build rerankers from application Settings.

The package-level entry points :func:`reranker`, :func:`areranker` and
:func:`transform` dispatch to a named implementation via the
``method`` keyword.

Long-context LLM rerank is exposed as :class:`Context` (was the
``LongContextRerankPass`` module — the "long_context" name was just
noise; the class is a context-bound rerank pass).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field, SecretStr

from raghub.config import LongContextConfig, Settings
from raghub.core import allowed_company_filter
from raghub.embedder import Embedder
from raghub.errors import GraphUnavailableError, RerankerError
from raghub.models import (
    Chunk,
    Classification,
    Hit,
    RankedList,
    Turn,
    User,
)
from raghub.telemetry import record_long_context, record_rerank_latency
from raghub.utils import capture

VariantKind = Literal["original", "hyde", "multi_query", "step_back", "sub"]


class Variant(BaseModel):
    """A single rephrased question ready for retrieval.

    Attributes:
        text: The rewritten question.
        kind: Discriminator string for telemetry.
        weight: Multiplier applied when the variant's hits are fused.
            ``1.5`` biases retrieval toward the user's literal wording.

    """

    text: str
    kind: VariantKind = "original"
    weight: float = Field(default=1.0, ge=0.0)


@runtime_checkable
class Rerank(Protocol):
    """A reranker: reorder retrieval hits using a downstream signal.

    Implementations can be sync only (``rerank``), async only (or wrap a
    sync model with ``asyncio.run`` to expose ``arerank``), or both.
    """

    name: str

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Rerank ``hits`` for ``question`` synchronously; may block."""
        ...

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Asynchronously rerank ``hits`` for ``question``."""
        ...


@runtime_checkable
class Transformer(Protocol):
    """Async rewriter turning a question into multiple variants.

    Attributes:
        name: Stable identifier used for telemetry and config.

    """

    name: str

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn],
    ) -> list[Variant]:
        """Return rephrased variants for ``question``."""
        ...


ORIGINAL_WEIGHT = 1.5


# ---------------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------------


class Identity:
    """No-op reranker.

    Attributes:
        name: ``"identity"``.

    """

    name = "identity"

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Return ``hits`` unchanged (identity pass-through)."""
        return list(hits)

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async identity pass-through."""
        return list(hits)


def rerank_latency(provider: str, seconds: float) -> None:
    """Push a histogram observation when Prometheus is wired up."""
    record_rerank_latency(provider, seconds)


class Cohere:
    """Cohere cross-encoder reranker.

    Attributes:
        name: ``"cohere"``.

    """

    name = "cohere"

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        *,
        model: str = "rerank-english-v3.0",
        top_k: int = 20,
        client: Any | None = None,
    ) -> None:
        """Initialise the reranker.

        Args:
            api_key: Cohere API key. Defaults to ``COHERE_API_KEY`` env var.
            model: Cohere rerank model name.
            top_k: Maximum candidates scored.
            client: Optional pre-built :class:`cohere.Client` (skips init).

        Raises:
            RerankerError: When no API key is available.

        """
        resolved = api_key
        if resolved is None:
            env = os.getenv("COHERE_API_KEY")
            if not env:
                raise RerankerError("Cohere requires COHERE_API_KEY or an explicit api_key")
            resolved = env
        self.api_key = resolved if isinstance(resolved, SecretStr) else SecretStr(resolved)
        self.model = model
        self.top_k = top_k
        self.client = client

    def ensure_client(self) -> Any:
        """Return the underlying :class:`cohere.Client`."""
        if self.client is None:
            import cohere

            self.client = cohere.Client(api_key=self.api_key.get_secret_value())
        return self.client

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Reorder ``hits`` by Cohere's relevance score."""
        if not hits:
            return []
        started = time.perf_counter()
        ordered = self.score(question, hits)
        rerank_latency(self.name, time.perf_counter() - started)
        return ordered

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async shim that pushes the sync rerank onto a worker thread."""
        return cast(
            list[Hit],
            await __import__("asyncio").to_thread(self.rerank, question=question, hits=list(hits)),
        )

    def score(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Call the Cohere API and reorder ``hits`` by its output."""
        client = self.ensure_client()
        documents = [hit.chunk.text for hit in hits]
        response = client.rerank(
            model=self.model,
            query=question,
            documents=documents,
            top_n=min(self.top_k, len(documents)),
        )
        ordered: list[Hit] = []
        for result in getattr(response, "results", []):
            idx = getattr(result, "index", None)
            if idx is None or idx < 0 or idx >= len(hits):
                continue
            ordered.append(hits[idx])
        return ordered


class Cascade:
    """Two-stage reranker: ``cheap`` then ``expensive`` (conditionally).

    The expensive reranker is invoked only when the cheap reranker did
    not reorder the input list — i.e. cheap "didn't have an opinion".

    Attributes:
        name: ``"cascade"``.

    """

    name = "cascade"

    def __init__(
        self,
        cheap: Any,
        expensive: Any,
        *,
        spread_threshold: float = 0.05,
    ) -> None:
        """Initialise the cascade.

        Args:
            cheap: First-stage reranker (sync or async).
            expensive: Second-stage reranker invoked only when cheap
                returned the input unchanged.
            spread_threshold: Reserved for future use when cheap
                rerankers expose confidence.

        """
        self.cheap = cheap
        self.expensive = expensive
        self.spread_threshold = float(spread_threshold)

    @staticmethod
    def changed_order(input_hits: Sequence[Hit], ranked: Sequence[Hit]) -> bool:
        """Return ``True`` when ``ranked`` is not the input order."""
        if len(input_hits) != len(ranked):
            return True
        return [h.chunk_id for h in input_hits] != [h.chunk_id for h in ranked]

    @staticmethod
    async def call(reranker: Any, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Call ``arerank`` when available, else ``rerank`` in a thread."""
        arerank = getattr(reranker, "arerank", None)
        if callable(arerank):
            return list(await arerank(question=question, hits=list(hits)))
        sync = getattr(reranker, "rerank", None)
        if callable(sync):
            return cast(
                list[Hit],
                await __import__("asyncio").to_thread(sync, question=question, hits=list(hits)),
            )
        raise TypeError(f"reranker {reranker!r} has neither arerank nor rerank")

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async cascade.

        Returns ``cheap.rerank(hits)`` when cheap reordered, else
        ``expensive.rerank(cheap.rerank(hits))``.
        """
        if not hits:
            return []
        cheap_ranked = await self.call(self.cheap, question, hits)
        if self.changed_order(hits, cheap_ranked):
            return list(cheap_ranked)
        expensive_ranked = await self.call(self.expensive, question, cheap_ranked)
        id_to_hit = {h.chunk_id: h for h in cheap_ranked}
        ordered = [id_to_hit.get(h.chunk_id, h) for h in expensive_ranked]
        ordered_set = {h.chunk_id for h in ordered}
        for h in cheap_ranked:
            if h.chunk_id not in ordered_set:
                ordered.append(h)
        return ordered

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Sync shim around :meth:`arerank`."""
        return cast(
            list[Hit],
            __import__("asyncio").run(self.arerank(question=question, hits=hits)),
        )


LISTWISE_MAX = 10


def extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Pull the first JSON array of objects out of (possibly fenced) text."""
    if not raw:
        return []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("[")
    if start == -1:
        return []
    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return []
    parsed, _ = capture(json.loads, candidate[start:end])
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def merge_with_rrf(per_window: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Reciprocal-Rank-Fusion merge across ranked windows."""
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranked in per_window:
        for rank, hit in enumerate(ranked, start=1):
            cid = hit.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in order:
                order[cid] = len(order)
    return sorted(
        {hit.chunk_id: hit for window in per_window for hit in window}.values(),
        key=lambda h: (-scores.get(h.chunk_id, 0.0), order.get(h.chunk_id, 0)),
    )


class LlmJudge:
    """LLM-as-judge listwise / pairwise reranker.

    Attributes:
        name: ``"llm"``.

    """

    name = "llm"

    def __init__(
        self,
        *,
        llm: Any,
        top_k: int = 20,
    ) -> None:
        """Initialise the reranker.

        Args:
            llm: Object with ``async_generate``.
            top_k: Maximum candidates scored.

        """
        self.llm = llm
        self.top_k = top_k

    async def rank_window(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Listwise-rank a single window of candidates."""
        lines = []
        for idx, hit in enumerate(hits):
            snippet = (hit.chunk.text or "").replace("\n", " ")[:400]
            lines.append(f"[{idx}] {snippet}")
        prompt = (
            "Rank the following passages by relevance to the question.\n"
            f"Question: {question}\n\n"
            "Passages:\n"
            + "\n".join(lines)
            + '\n\nReturn a JSON array of objects [{"index": <int>, "score": <0..1>}] '
            "sorted by descending score. No prose, no markdown."
        )
        raw = await self.llm.async_generate(
            system_prompt="You rank passages for retrieval relevance.",
            conversation=[],
            context=[],
            question=prompt,
        )
        parsed = extract_json_array(raw or "")
        ordered: list[Hit] = []
        seen: set[int] = set()
        for item in parsed:
            index_value: Any = item.get("index")
            if (
                not isinstance(index_value, int)
                or index_value < 0
                or index_value >= len(hits)
                or index_value in seen
            ):
                continue
            ordered.append(hits[index_value])
            seen.add(index_value)
        for idx, hit in enumerate(hits):
            if idx not in seen:
                ordered.append(hit)
        return ordered

    async def do_rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Listwise for ≤ LISTWISE_MAX candidates; windowed RRF above."""
        if len(hits) <= LISTWISE_MAX:
            return (await self.rank_window(question, hits))[: self.top_k]
        windows = [hits[i : i + LISTWISE_MAX] for i in range(0, len(hits), LISTWISE_MAX)]
        per_window = [await self.rank_window(question, w) for w in windows]
        merged = merge_with_rrf(per_window)
        return merged[: self.top_k]

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async rerank."""
        if not hits:
            return []
        started = time.perf_counter()
        ordered = await self.do_rerank(question, list(hits))
        rerank_latency(self.name, time.perf_counter() - started)
        return ordered

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Sync rerank via :func:`asyncio.run`."""
        return cast(
            list[Hit],
            __import__("asyncio").run(self.arerank(question=question, hits=hits)),
        )


CONTEXT = (
    "You re-rank retrieved passages. For every candidate, produce a "
    "relevance score in [0, 1] and a one-sentence rationale. Reply "
    "with JSON only — no prose, no markdown."
)


def context_prompt(question: str, hits: Sequence[Hit]) -> str:
    """Assemble the long-context prompt."""
    lines = [f"Question: {question}", "", "Candidates:"]
    for idx, hit in enumerate(hits):
        snippet = (hit.chunk.text or "").replace("\n", " ")[:600]
        lines.append(f"[{idx}] id={hit.chunk_id} text={snippet}")
    lines.append("")
    lines.append(
        "Return a JSON object: "
        '{"items": [{"chunk_id": "<id>", "score": <0..1>, '
        '"rationale": "<one sentence>"}, ...]} '
        "ordered by descending score. No prose."
    )
    return "\n".join(lines)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a (possibly fenced) string."""
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


def reorder_candidates(
    candidates: Sequence[Hit],
    ranked: RankedList,
) -> list[Hit] | None:
    """Apply the LLM's ranking to ``candidates``."""
    id_to_hit = {hit.chunk_id: hit for hit in candidates}
    ordered: list[Hit] = []
    seen: set[str] = set()
    for item in ranked.items:
        if item.chunk.id in id_to_hit and item.chunk.id not in seen:
            ordered.append(id_to_hit[item.chunk.id])
            seen.add(item.chunk.id)
    if not ordered:
        return None
    for hit in candidates:
        if hit.chunk_id not in seen:
            ordered.append(hit)
    return ordered


def record_context_latency(outcome: str, seconds: float) -> None:
    """Push a long-context counter observation when Prometheus is wired."""
    record_long_context(outcome=outcome, seconds=seconds)


class Context:
    """Long-context second-pass reranker.

    Was named ``LongContextRerankPass``. The "long_context" prefix
    was redundant — context is what this thing consumes.

    Attributes:
        name: ``"long_context"``.

    """

    name = "long_context"

    def __init__(self, llm: Any, settings: LongContextConfig) -> None:
        """Initialise the pass.

        Args:
            llm: LLM provider with an ``async_generate`` method.
            settings: The :class:`LongContextConfig` block.

        """
        self.llm = llm
        self.settings = settings

    def is_eligible(self) -> bool:
        """Return ``True`` when the pass should run for the current LLM."""
        if not self.settings.enabled:
            return False
        model_name = getattr(self.llm, "model_name", "") or ""
        if not model_name:
            return False
        return model_name in (self.settings.allowlist_models or [])

    async def rerank(
        self,
        *,
        question: str,
        hits: Sequence[Hit],
    ) -> list[Hit]:
        """Re-order ``hits`` with a long-context LLM call.

        Returns the original order when the pass is not eligible,
        the LLM errors, or the response cannot be parsed.
        """
        if not self.is_eligible() or not hits:
            return list(hits)
        candidates = list(hits[: max(1, self.settings.candidate_k)])
        started = time.perf_counter()
        try:
            raw = await self.llm.async_generate(
                system_prompt=CONTEXT,
                conversation=[],
                context=[],
                question=context_prompt(question, candidates),
            )
            parsed = extract_json_object(raw or "")
            if parsed is None:
                record_context_latency("bad_json", time.perf_counter() - started)
                return list(hits)
            ranked = RankedList.model_validate(parsed)
            reordered = reorder_candidates(candidates, ranked)
            if reordered is None:
                record_context_latency("bad_json", time.perf_counter() - started)
                return list(hits)
            record_context_latency("ran", time.perf_counter() - started)
            return reordered
        except Exception:  # pragma: no cover - defensive envelope
            record_context_latency("error", time.perf_counter() - started)
            return list(hits)

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async alias preserved for symmetry with other rerankers."""
        return await self.rerank(question=question, hits=hits)


class Colbert:
    """Adapter for the optional :mod:`ragatouille` ColBERT backend."""

    name = "colbert"

    def __init__(self, config: Any | None = None) -> None:
        """Initialise the adapter.

        Args:
            config: Optional :class:`HybridConfig` carrying the
                ``colbert_enabled`` flag. ``None`` defaults to disabled.

        """
        self.config = config
        self.enabled = bool(getattr(config, "colbert_enabled", False))
        self.index: Any | None = None

    def is_available(self) -> bool:
        """Return ``True`` when ColBERT is enabled and importable."""
        if not self.enabled:
            return False
        import importlib.util

        return importlib.util.find_spec("ragatouille") is not None

    def score(self, query: str, doc_texts: list[str]) -> list[float]:
        """Return ColBERT relevance scores parallel to ``doc_texts``.

        Raises:
            GraphUnavailableError: When ``colbert_enabled`` is ``True``
                but the dependency is missing.

        """
        if not doc_texts:
            return []
        if not self.is_available():
            if self.enabled:
                raise GraphUnavailableError(
                    "colbert_enabled is True but ragatouille is not installed; "
                    "pip install 'raghub[colbert]' to enable ColBERT late-interaction"
                )
            return []
        from ragatouille import RAGPretrainedModel

        if self.index is None:
            self.index = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        return list(self.index.rerank(query=query, documents=doc_texts))


# ---------------------------------------------------------------------------
# Fusion / ranking combination
# ---------------------------------------------------------------------------


class Fusion:
    """Fuse ranked result lists using RRF or weighted linear scoring."""

    def __init__(self, method: str = "rrf", *, k: int = 60) -> None:
        """Initialise fusion with ``method`` and RRF damping constant.

        Args:
            method: ``"rrf"`` (default) or ``"linear"``.
            k: RRF damping constant.

        Raises:
            ValueError: When ``method`` is unknown or ``k < 1``.

        """
        if method not in {"rrf", "linear"}:
            raise ValueError(f"Unknown fusion method: {method!r}")
        if k < 1:
            raise ValueError("rrf k must be >= 1")
        self.method = method
        self.k = k

    def fuse(self, lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse lists of result dictionaries keyed by ``chunk_id``."""
        if self.method == "rrf":
            scores: dict[str, float] = {}
            records: dict[str, dict[str, Any]] = {}
            for ranking in lists:
                for rank, item in enumerate(ranking, start=1):
                    chunk_id = item["chunk_id"]
                    if not isinstance(chunk_id, str):
                        raise ValueError(f"chunk_id must be str, got {type(chunk_id).__name__}")
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.k + rank)
                    records.setdefault(chunk_id, item)
            return [
                records[key] | {"score": score}
                for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]
        linear_scores: dict[str, float] = {}
        records_linear: dict[str, dict[str, Any]] = {}
        for ranking in lists:
            maximum = max((float(item.get("score", 0)) for item in ranking), default=0.0) or 1.0
            for item in ranking:
                key = item["chunk_id"]
                if not isinstance(key, str):
                    raise ValueError(f"chunk_id must be str, got {type(key).__name__}")
                linear_scores[key] = (
                    linear_scores.get(key, 0.0) + float(item.get("score", 0)) / maximum
                )
                records_linear.setdefault(key, item)
        return [
            records_linear[key] | {"score": score}
            for key, score in sorted(linear_scores.items(), key=lambda x: x[1], reverse=True)
        ]


def rrf(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse ordered chunk-id rankings with reciprocal rank fusion.

    Args:
        rankings: Each inner list is a ranked channel of chunk ids.
        k: RRF damping constant.

    Returns:
        A list of ``(chunk_id, score)`` tuples sorted by descending
        fused score.

    Raises:
        ValueError: When ``k < 1`` or any chunk id is not a ``str``.

    """
    rows: list[list[dict[str, Any]]] = []
    for ranking in rankings:
        rows.append([{"chunk_id": item} for item in ranking])
    fused = Fusion(k=k).fuse(rows)
    return [(row["chunk_id"], float(row["score"])) for row in fused]


def linear_combine(
    channel_scores: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Combine max-normalised channel scores.

    Args:
        channel_scores: ``{channel_name: {chunk_id: score}}`` mapping.
        weights: Optional per-channel multiplier AFTER per-channel
            max normalisation.

    Returns:
        A list of ``(chunk_id, fused_score)`` tuples sorted by descending score.

    """
    weight_map = dict(weights or {})
    scores: dict[str, float] = {}
    for channel_name, channel in channel_scores.items():
        if not channel:
            continue
        max_score = max(channel.values())
        if max_score <= 0:
            max_score = 1.0
        weight = weight_map.get(channel_name, 1.0)
        for chunk_id, raw in channel.items():
            normalised = float(raw) / max_score
            scores[chunk_id] = scores.get(chunk_id, 0.0) + normalised * weight
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Query transformers
# ---------------------------------------------------------------------------


def extract_string_array(raw: str) -> list[str]:
    """Pull a JSON array of strings out of a (possibly fenced) string."""
    if not raw:
        return []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("[")
    if start == -1:
        return []
    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return []
    parsed, _ = capture(json.loads, candidate[start:end])
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


HYDE = (
    "You generate hypothetical passages for retrieval. Reply with the "
    "passage only — no preamble, no heading, no commentary."
)


def hyde_prompt(question: str) -> str:
    """Build the HyDE prompt that elicits a hypothetical passage for retrieval."""
    return (
        "Write a short paragraph (3-5 sentences) that would answer the "
        "following question. The paragraph does not need to be factual — "
        "it just needs to use the same vocabulary and phrasing a real "
        "source document would use.\n\n"
        f"Question: {question}\n\nPassage:"
    )


class Hyde:
    """HyDE (Hypothetical Document Embeddings) transformer.

    Attributes:
        name: Always ``"hyde"``.

    """

    name = "hyde"

    def __init__(self, llm: Any, *, n: int = 1) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with an ``async_generate`` method.
            n: Number of hypothetical passages to generate.

        """
        if n < 1:
            raise ValueError("HyDE ``n`` must be >= 1")
        self.llm = llm
        self.n = n

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Generate ``n`` hypothetical passages."""
        prompt = hyde_prompt(question)
        variants: list[Variant] = []
        for _ in range(self.n):
            text = await self.llm.async_generate(
                system_prompt=HYDE,
                conversation=list(history),
                context=[],
                question=prompt,
            )
            text = (text or "").strip()
            if text:
                variants.append(Variant(text=text, kind="hyde"))
        return variants


MULTI_QUERY = (
    "You generate alternative phrasings of a question for retrieval. "
    "Reply with a JSON array of strings only — no prose, no preamble."
)


def multi_query_prompt(question: str, n: int) -> str:
    """Build the multi-query prompt that returns N rephrasings as JSON."""
    return (
        f"Rewrite the following question as {n} distinct search queries. "
        "Vary vocabulary and structure; keep the intent identical. "
        "Output a JSON array of strings.\n\n"
        f"Question: {question}\n\nJSON:"
    )


class MultiQuery:
    """Multi-query rewriter.

    Attributes:
        name: Always ``"multi_query"``.

    """

    name = "multi_query"

    def __init__(self, llm: Any, *, n: int = 4) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.
            n: Number of rephrasings to request.

        """
        if n < 1:
            raise ValueError("multi-query ``n`` must be >= 1")
        self.llm = llm
        self.n = n

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Generate ``n`` alternative phrasings."""
        raw = await self.llm.async_generate(
            system_prompt=MULTI_QUERY,
            conversation=list(history),
            context=[],
            question=multi_query_prompt(question, self.n),
        )
        phrasings = extract_string_array(raw or "")
        return [Variant(text=phrase, kind="multi_query") for phrase in phrasings[: self.n]]


DECOMPOSE = (
    "You decompose compound questions into independent sub-questions "
    "for retrieval. Reply with a JSON array of strings only — no prose."
)


def decompose_prompt(question: str) -> str:
    """Build the decomposition prompt that splits a compound question."""
    return (
        "Split the following compound question into the minimum set of "
        "independent sub-questions whose answers together imply the "
        "original answer. Output a JSON array of strings.\n\n"
        f"Question: {question}\n\nJSON:"
    )


class Decompose:
    """Decomposition transformer.

    Attributes:
        name: Always ``"decompose"``.

    """

    name = "decompose"

    def __init__(self, llm: Any) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.

        """
        self.llm = llm

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Produce sub-question variants."""
        raw = await self.llm.async_generate(
            system_prompt=DECOMPOSE,
            conversation=list(history),
            context=[],
            question=decompose_prompt(question),
        )
        sub_questions = extract_string_array(raw or "")
        return [Variant(text=q, kind="sub") for q in sub_questions]


STEP_BACK = (
    "You reframe a specific question as a more abstract, principle-"
    "level question. Reply with one sentence only — no preamble."
)


def step_back_prompt(question: str) -> str:
    """Build the step-back prompt that asks for the principle-level question."""
    return (
        "Given the specific question below, write the more general, "
        "principle-level question that would provide useful background. "
        "Reply with one sentence only.\n\n"
        f"Specific: {question}\n\nAbstract:"
    )


class StepBack:
    """Step-back prompting transformer.

    Attributes:
        name: Always ``"step_back"``.

    """

    name = "step_back"

    def __init__(self, llm: Any) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.

        """
        self.llm = llm

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Produce the abstract reformulation."""
        abstract = await self.llm.async_generate(
            system_prompt=STEP_BACK,
            conversation=list(history),
            context=[],
            question=step_back_prompt(question),
        )
        text = (abstract or "").strip()
        if not text:
            return []
        return [Variant(text=text, kind="step_back", weight=1.2)]


class Compose:
    """Run several transforms in order; prepend the original question.

    The original question is always present in the output (weight
    ``1.5``) so retrieval is biased toward the user's literal phrasing
    even when every transform fails or returns nothing.

    Attributes:
        name: ``"compose"``.

    """

    name = "compose"

    def __init__(self, transformers: Sequence[Transformer]) -> None:
        """Initialise the composer.

        Args:
            transformers: Ordered list of transforms to apply. Each is
                awaited sequentially.

        """
        self.transformers: list[Transformer] = list(transformers)

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Combine the original question with every transformer's output."""
        variants: list[Variant] = [Variant(text=question, kind="original", weight=ORIGINAL_WEIGHT)]
        for t in self.transformers:
            produced = await t.transform(question=question, history=list(history))
            variants.extend(produced)
        return variants


# ---------------------------------------------------------------------------
# Faceted search
# ---------------------------------------------------------------------------


@dataclass
class SearchFilters:
    """Filter criteria for faceted search.

    Attributes:
        companies: Allowed company tags.
        departments: Allowed department tags.
        classifications: Allowed document classifications.
        owners: Allowed owner emails.
        date_from: Lower bound for document date (inclusive).
        date_to: Upper bound for document date (inclusive).
        file_types: Allowed file extensions.

    """

    companies: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    file_types: list[str] = field(default_factory=list)


def build_filter(filters: SearchFilters | None) -> str:
    """Serialize :class:`SearchFilters` to a SQL-style metadata filter string."""
    if filters is None:
        return ""
    clauses: list[str] = []
    if filters.companies:
        quoted = ", ".join(f"'{c}'" for c in filters.companies)
        clauses.append(f"company IN ({quoted})")
    if filters.owners:
        quoted = ", ".join(f"'{o}'" for o in filters.owners)
        clauses.append(f"owner IN ({quoted})")
    if filters.file_types:
        quoted = ", ".join(f"'{t}'" for t in filters.file_types)
        clauses.append(f"file_type IN ({quoted})")
    return " AND ".join(clauses)


class Search:
    """Advanced search with faceted filtering for chunks."""

    def __init__(self, vector_store: Any, embedding_provider: Any) -> None:
        """Initialise the search engine.

        Args:
            vector_store: A :class:`VectorStore`-conforming instance.
            embedding_provider: An :class:`EmbeddingProvider`-conforming instance.

        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list[Chunk]:
        """Search with faceted filtering."""
        vector = self.embedding_provider.embed_text(query)
        metadata_filter = build_filter(filters)
        raw = self.vector_store.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
        results: list[Chunk] = []
        seen: set[str] = set()
        for item in raw:
            chunk: Chunk = item["chunk"]
            if chunk.id in seen:
                continue
            if filters and not self.matches(chunk, filters):
                continue
            seen.add(chunk.id)
            results.append(chunk)
        return results

    @staticmethod
    def matches(chunk: Chunk, filters: SearchFilters) -> bool:
        """Return ``True`` when ``chunk`` satisfies every active filter criterion."""
        return (
            chunk.company in (filters.companies or [chunk.company])
            and chunk.department in (filters.departments or [chunk.department])
            and chunk.classification in (filters.classifications or [chunk.classification])
            and chunk.owner in (filters.owners or [chunk.owner])
        )

    def count_by_field(self, field: str) -> dict[str, int]:
        """Return facet counts for a given metadata field."""
        records = getattr(self.vector_store, "records", None)
        if records is None:
            return {}
        counts: dict[str, int] = {}
        for rec in records.values():
            value = getattr(rec.chunk, field, None)
            if value is None:
                continue
            if isinstance(value, list):
                for v in value:
                    counts[v] = counts.get(v, 0) + 1
            else:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Pipeline + reranker factory
# ---------------------------------------------------------------------------


class Retrieval:
    """Vector + keyword retrieval with deduplication and weighted fusion.

    Attributes:
        embedding_provider: Embeds queries into the same vector space.
        vector_store: Performs the actual vector + keyword searches.
        rerank: Optional reranker applied after dedupe. The default
            :class:`Identity` is a no-op.

    """

    def __init__(
        self,
        *,
        embedding_provider: Embedder,
        vector_store: Any,
        rerank: Rerank,
        hybrid: Any | None = None,
    ) -> None:
        """Wire the pipeline to its collaborators.

        Args:
            embedding_provider: Used to embed incoming queries.
            vector_store: Performs vector and keyword searches.
            rerank: Applied after dedupe to reorder hits.
            hybrid: Hybrid-retrieval fusion config (defaults to RRF, k=60).

        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.rerank = rerank
        self.hybrid = hybrid or default_hybrid_config()

    def retrieve(self, *, user: User, question: str, top_k: int) -> list[Hit]:
        """Retrieve authorised, deduplicated chunks relevant to ``question``."""
        metadata_filter = allowed_company_filter(user)
        vector = self.embedding_provider.embed_text(question)
        raw_hits = self.vector_store.search(
            vector=vector, top_k=top_k, metadata_filter=metadata_filter
        )
        hits: list[Hit] = []
        seen: set[str] = set()
        for raw in raw_hits:
            chunk: Chunk = raw["chunk"]
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            hits.append(
                Hit(
                    score=float(raw["score"]),
                    chunk=chunk,
                )
            )
        return self.rerank.rerank(question=question, hits=hits)

    def retrieve_keyword(self, query: str, top_k: int = 5) -> list[Hit]:
        """Keyword-only retrieval using the vector store's native scorer."""
        raw_hits = self.vector_store.keyword_search(query, top_k)
        return [
            Hit(
                score=float(h["score"]),
                chunk=h["chunk"],
            )
            for h in raw_hits
        ]

    def retrieve_hybrid(
        self,
        query: str,
        vector_results: list[Hit],
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        *,
        fusion: str | None = None,
        rrf_k: int | None = None,
    ) -> list[Hit]:
        """Combine keyword and vector hits with the configured fusion."""
        chosen = fusion or getattr(self.hybrid, "fusion", "rrf")
        if chosen == "linear":
            return self.linear(
                query=query,
                vector_results=vector_results,
                keyword_weight=keyword_weight,
                vector_weight=vector_weight,
            )
        return self.fused(
            query=query,
            vector_results=vector_results,
            rrf_k=rrf_k if rrf_k is not None else getattr(self.hybrid, "rrf_k", 60),
        )

    def fused(
        self,
        *,
        query: str,
        vector_results: list[Hit],
        rrf_k: int,
    ) -> list[Hit]:
        """Reciprocal-Rank-Fusion hybrid path."""
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2 or 1)
        dense_ranks = [h.chunk_id for h in vector_results]
        sparse_ranks = [h.chunk_id for h in keyword_hits]
        fused = rrf([dense_ranks, sparse_ranks], k=rrf_k)
        chunk_map: dict[str, Chunk] = {h.chunk_id: h.chunk for h in keyword_hits}
        chunk_map.update({h.chunk_id: h.chunk for h in vector_results})
        out: list[Hit] = []
        for chunk_id, score in fused:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                out.append(Hit(score=float(score), chunk=chunk))
        return out

    def linear(
        self,
        *,
        query: str,
        vector_results: list[Hit],
        keyword_weight: float,
        vector_weight: float,
    ) -> list[Hit]:
        """Legacy linear-combine path."""
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2 or 1)
        keyword_by_id: dict[str, float] = {h.chunk_id: h.score for h in keyword_hits}
        vector_by_id: dict[str, float] = {h.chunk_id: h.score for h in vector_results}
        all_ids = set(keyword_by_id) | set(vector_by_id)
        kw_max = max(keyword_by_id.values()) if keyword_by_id else 1.0
        vec_max = (
            max(vector_by_id.values()) if vector_by_id and max(vector_by_id.values()) > 0 else 1.0
        )
        chunk_map: dict[str, Chunk] = {}
        for h in keyword_hits:
            chunk_map[h.chunk_id] = h.chunk
        for h in vector_results:
            chunk_map[h.chunk_id] = h.chunk
        fused: list[Hit] = []
        for chunk_id in all_ids:
            kw_score = keyword_by_id.get(chunk_id, 0.0) / kw_max
            vec_score = vector_by_id.get(chunk_id, 0.0) / vec_max
            combined = keyword_weight * kw_score + vector_weight * vec_score
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                fused.append(Hit(score=combined, chunk=chunk))
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused

    def hybrid_search(self, *, user: User, question: str, top_k: int) -> list[Hit]:
        """RBAC-filtered vector hits fused with keyword hits."""
        vector_results = self.retrieve(user=user, question=question, top_k=top_k)
        return self.retrieve_hybrid(question, vector_results)

    def retrieve_variants(
        self,
        *,
        user: User,
        variants: list[Variant],
        top_k: int,
    ) -> list[Hit]:
        """Embed each variant and fuse the per-channel maxima into one ranking."""
        if not variants:
            return []
        if (
            len(variants) == 1
            and variants[0].kind == "original"
            and getattr(variants[0], "text", "") != ""
        ):
            return self.retrieve(user=user, question=variants[0].text, top_k=top_k)
        chunk_score: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        for variant in variants:
            text = (getattr(variant, "text", "") or "").strip()
            if not text:
                continue
            weight = float(getattr(variant, "weight", 1.0) or 1.0)
            if weight <= 0:
                continue
            hits = self.retrieve(user=user, question=text, top_k=top_k)
            if not hits:
                continue
            channel_max = max((h.score for h in hits), default=1.0) or 1.0
            for hit in hits:
                contribution = (hit.score / channel_max) * weight
                prior = chunk_score.get(hit.chunk_id, 0.0)
                if contribution > prior:
                    chunk_score[hit.chunk_id] = contribution
                chunk_map.setdefault(hit.chunk_id, hit.chunk)
        fused: list[Hit] = [
            Hit(score=score, chunk=chunk_map[cid])
            for cid, score in chunk_score.items()
            if cid in chunk_map
        ]
        fused.sort(key=lambda h: h.score, reverse=True)
        question_for_rerank = next(
            (getattr(v, "text", "") for v in variants if getattr(v, "text", "")),
            "",
        )
        return self.rerank.rerank(question=question_for_rerank, hits=fused)

    def retrieve_hybrid_v2(
        self,
        *,
        user: User,
        question: str,
        top_k: int,
        colbert: Any | None = None,
    ) -> list[Hit]:
        """Three-channel hybrid retrieval (dense + sparse + optional ColBERT)."""
        dense = self.retrieve(user=user, question=question, top_k=top_k)
        sparse = self.retrieve_keyword(question, top_k=top_k)
        colbert_hits: list[Hit] = []
        if colbert is not None and getattr(colbert, "is_available", lambda: False)():
            scores = colbert.score(question, [h.chunk.text for h in dense])
            if scores and len(scores) == len(dense):
                colbert_hits = [
                    Hit(
                        score=float(score), chunk=h.chunk,
                    )
                    for h, score in zip(dense, scores, strict=True)
                ]
        rankings = [
            [h.chunk_id for h in dense],
            [h.chunk_id for h in sparse],
            [h.chunk_id for h in colbert_hits],
        ]
        rankings = [r for r in rankings if r]
        if not rankings:
            return []
        fused_scores = rrf(rankings, k=getattr(self.hybrid, "rrf_k", 60))
        chunk_map: dict[str, Chunk] = {h.chunk_id: h.chunk for h in sparse}
        chunk_map.update({h.chunk_id: h.chunk for h in colbert_hits})
        chunk_map.update({h.chunk_id: h.chunk for h in dense})
        out: list[Hit] = []
        for chunk_id, score in fused_scores:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                out.append(Hit(score=float(score), chunk=chunk))
        return self.rerank.rerank(question=question, hits=out)


class RerankerFactory:
    """Build reranker instances from application settings."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any | None = None,
        cohere_api_key: str | None = None,
    ) -> None:
        """Initialise the factory dependencies."""
        self.settings = settings
        self.llm = llm
        self.cohere_api_key = cohere_api_key

    def create(self, spec: str | None = None) -> Rerank:
        """Create the configured (or named) reranker."""
        cfg = self.settings.reranker
        provider = spec or cfg.provider
        if provider == "none" or provider == "identity":
            return Identity()
        if provider == "cohere":
            return Cohere(
                api_key=self.cohere_api_key,
                model="rerank-english-v3.0",
                top_k=cfg.top_k,
            )
        if provider == "llm":
            return LlmJudge(llm=self.llm, top_k=cfg.top_k)
        if provider == "cascade":
            expensive: Rerank
            if self.cohere_api_key is None and not os.getenv("COHERE_API_KEY"):
                return Identity()
            expensive = Cohere(api_key=self.cohere_api_key, top_k=cfg.top_k)
            return Cascade(
                cheap=Identity(),
                expensive=expensive,
                spread_threshold=cfg.cascade_threshold,
            )
        if provider == "long_context":
            if self.llm is None:
                raise RerankerError(
                    "long_context reranker requires an LLM via RerankerFactory(llm=...)"
                )
            # Context is async-only; Rerank requires sync rerank, so callers
            # reach it through the async path or asyncio.run (see reranker()).
            return cast(
                Rerank,
                Context(self.llm, getattr(cfg, "long_context", None) or default_long_context()),
            )
        raise RerankerError(f"Unknown reranker provider: {provider!r}")


def build_reranker(
    settings: Settings,
    *,
    llm: Any | None = None,
    cohere_api_key: str | None = None,
) -> Rerank:
    """Build the configured reranker."""
    return RerankerFactory(settings, llm=llm, cohere_api_key=cohere_api_key).create()


# ---------------------------------------------------------------------------
# Default config helpers
# ---------------------------------------------------------------------------


def default_hybrid_config() -> Any:
    """Construct a default ``HybridConfig`` (or duck-typed stand-in)."""
    try:
        from raghub.config import HybridConfig

        return HybridConfig()
    except Exception:
        return HybridConfigShim()


class HybridConfigShim:
    """Last-resort shim if the real config class is unavailable."""

    fusion = "rrf"
    rrf_k = 60
    colbert_enabled = False
    long_context: Any = None


def default_long_context() -> LongContextConfig:
    """Return a disabled :class:`LongContextConfig` for fallback construction."""
    return LongContextConfig(enabled=False, candidate_k=5, allowlist_models=[])


# ---------------------------------------------------------------------------
# Top-level dispatcher functions used by callers and the CLI
# ---------------------------------------------------------------------------


def reranker(
    question: str,
    hits: Sequence[Hit],
    *,
    method: str = "identity",
) -> list[Hit]:
    """Rerank ``hits`` synchronously using the named ``method``.

    Args:
        question: The user query.
        hits: The candidate hits to reorder.
        method: Reranker name. One of ``identity``, ``cohere``,
            ``llm``, ``cascade``, ``long_context``.

    Returns:
        The hits reordered by descending relevance.

    """
    impl = reranker_from_method(method)
    out = impl.rerank(question=question, hits=list(hits))
    if __import__("asyncio").iscoroutine(out):
        return cast(
            list[Hit],
            __import__("asyncio").run(cast(Coroutine[Any, Any, list[Hit]], out)),
        )
    return out


async def areranker(
    question: str,
    hits: Sequence[Hit],
    *,
    method: str = "identity",
) -> list[Hit]:
    """Asynchronously rerank ``hits`` using the named ``method``."""
    impl = reranker_from_method(method)
    return await impl.arerank(question=question, hits=list(hits))


async def transform(
    question: str,
    history: Sequence[Turn] = (),
    *,
    method: str = "hyde",
    llm: Any | None = None,
) -> list[Variant]:
    """Asynchronously transform ``question`` using the named ``method``.

    Args:
        question: The user question.
        history: Recent in-window turns (defaults to empty).
        method: Transformer name. One of ``hyde``, ``multi_query``,
            ``decompose``, ``step_back``.
        llm: Object with ``async_generate`` (required for every method).

    Returns:
        The list of :class:`Variant`s produced.

    """
    if llm is None:
        raise RerankerError("transform(...) requires an LLM via llm=...")
    impl = transformer_from_method(method, llm)
    return await impl.transform(question=question, history=list(history))


def reranker_from_method(method: str) -> Rerank:
    """Construct a reranker by name. Settings-driven factory has its own path."""
    if method == "identity":
        return Identity()
    if method == "cohere":
        return Cohere()
    if method == "llm":
        from raghub.llm import LiteLLM

        return LlmJudge(llm=LiteLLM())
    if method == "cascade":
        return Cascade(cheap=Identity(), expensive=Identity())
    if method == "long_context":
        from raghub.llm import LiteLLM

        # Async-only (rerank is awaited by the pipeline); Rerank requires
        # a sync rerank, so callers reach it via arerank / asyncio.run.
        return cast(Rerank, Context(LiteLLM(), default_long_context()))
    raise RerankerError(f"Unknown reranker method: {method!r}")


def transformer_from_method(method: str, llm: Any) -> Transformer:
    """Construct a transformer by name."""
    if method == "hyde":
        return Hyde(llm)
    if method == "multi_query":
        return MultiQuery(llm)
    if method == "decompose":
        return Decompose(llm)
    if method == "step_back":
        return StepBack(llm)
    raise RerankerError(f"Unknown transform method: {method!r}")


__all__ = [
    "Cascade",
    "Cohere",
    "Colbert",
    "Compose",
    "Context",
    "Decompose",
    "Fusion",
    "Hyde",
    "Identity",
    "LlmJudge",
    "MultiQuery",
    "Rerank",
    "RerankerFactory",
    "Retrieval",
    "Search",
    "SearchFilters",
    "StepBack",
    "Transformer",
    "Variant",
    "areranker",
    "build_filter",
    "build_reranker",
    "context_prompt",
    "decompose_prompt",
    "extract_json_array",
    "extract_json_object",
    "extract_string_array",
    "hyde_prompt",
    "linear_combine",
    "merge_with_rrf",
    "multi_query_prompt",
    "record_context_latency",
    "reorder_candidates",
    "rerank_latency",
    "reranker",
    "rrf",
    "step_back_prompt",
    "transform",
]
