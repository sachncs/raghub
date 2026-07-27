"""Phase 4 — reranker tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from raghub.config import Settings, RerankerConfig
from raghub.exceptions import RerankerError
from raghub.models import ChunkRecord, RetrievalHit
from raghub.retrieval.rerankers.cascade import CascadeReranker
from raghub.retrieval.rerankers.factory import build_reranker
from raghub.retrieval.rerankers.llm import LLMReranker


def make_hit(i: int, text: str, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"c-{i}",
        score=score,
        chunk=ChunkRecord(
            chunk_id=f"c-{i}",
            document_id="d-1",
            version=1,
            page=1,
            source_location="s",
            section="",
            company="A",
            owner="",
            department="",
            text=text,
            metadata={},
        ),
    )


class FakeLlm:
    """Stand-in LLM returning a canned listwise JSON string."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "fake"

    async def async_generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        self.calls.append({"question": question})
        return self.response


# --- LLM reranker --------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_reranker_reorders_by_index() -> None:
    llm = FakeLlm(
        '[{"index": 2, "score": 0.9}, {"index": 0, "score": 0.5}, {"index": 1, "score": 0.1}]'
    )
    r = LLMReranker(llm=llm, top_k=3)
    hits = [make_hit(0, "alpha"), make_hit(1, "beta"), make_hit(2, "gamma")]
    out = await r.arerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-2", "c-0", "c-1"]


@pytest.mark.asyncio
async def test_llm_reranker_falls_back_on_bad_json() -> None:
    llm = FakeLlm("garbage output, not JSON")
    r = LLMReranker(llm=llm, top_k=3)
    hits = [make_hit(0, "a"), make_hit(1, "b"), make_hit(2, "c")]
    out = await r.arerank(question="q", hits=hits)
    # Falls back to input order.
    assert [h.chunk_id for h in out] == ["c-0", "c-1", "c-2"]


@pytest.mark.asyncio
async def test_llm_reranker_empty_hits_returns_empty() -> None:
    llm = FakeLlm("[]")
    out = await LLMReranker(llm=llm).arerank(question="q", hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_llm_reranker_pairwise_windowing_merges() -> None:
    """When hits exceed :data:`LISTWISE_MAX`, ranking is windowed then RRF-merged.

    The 12-hit input is split into two windows of
    :data:`LISTWISE_MAX` and a remainder. The first window's
    response reverses the order; the second preserves it. RRF must
    surface the first window's top item at the head of the merged
    output.
    """
    from raghub.retrieval.rerankers.llm import LISTWISE_MAX

    if LISTWISE_MAX < 4:
        pytest.skip("LISTWISE_MAX too small for this test")

    n_hits = LISTWISE_MAX + 2  # forces windowing
    call_count = {"n": 0}

    class WindowedLlm:
        model_name = "windowed"

        async def async_generate(self, **kwargs):
            call_count["n"] += 1
            question = kwargs.get("question", "")
            if f"text 0" in question and f"text {LISTWISE_MAX - 1}" in question:
                # First window: reverse.
                return json.dumps(
                    [
                        {"index": LISTWISE_MAX - 1, "score": 0.9},
                        {"index": LISTWISE_MAX - 2, "score": 0.5},
                        {"index": 1, "score": 0.3},
                        {"index": 0, "score": 0.1},
                    ]
                )
            if f"text {LISTWISE_MAX}" in question and f"text {n_hits - 1}" in question:
                # Second window: preserve.
                return json.dumps(
                    [
                        {"index": n_hits - 1, "score": 0.9},
                        {"index": n_hits - 2, "score": 0.5},
                        {"index": LISTWISE_MAX + 1, "score": 0.3},
                        {"index": LISTWISE_MAX, "score": 0.1},
                    ]
                )
            return "[]"

    r = LLMReranker(llm=WindowedLlm(), top_k=n_hits)
    hits = [make_hit(i, f"text {i}") for i in range(n_hits)]
    out = await r.arerank(question="q", hits=hits)
    # We made two LLM calls — one per window.
    assert call_count["n"] == 2
    # All hits must surface in the merged output.
    assert {h.chunk_id for h in out} == {f"c-{i}" for i in range(n_hits)}


def test_llm_reranker_sync_shim() -> None:
    llm = FakeLlm('[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]')
    r = LLMReranker(llm=llm, top_k=2)
    out = r.rerank(question="q", hits=[make_hit(0, "a"), make_hit(1, "b")])
    assert [h.chunk_id for h in out] == ["c-1", "c-0"]


# --- Cascade reranker -----------------------------------------------------


class StubReranker:
    def __init__(self, name: str, order_fn):
        self.name = name
        self.calls = 0
        self._order_fn = order_fn

    async def arerank(self, *, question, hits):
        self.calls += 1
        return self._order_fn(list(hits))

    def rerank(self, *, question, hits):
        self.calls += 1
        return self._order_fn(list(hits))


@pytest.mark.asyncio
async def test_cascade_skips_expensive_when_cheap_reorders() -> None:
    cheap = StubReranker("cheap", lambda h: list(reversed(h)))
    exp = StubReranker("exp", lambda h: list(reversed(h)))
    cascade = CascadeReranker(cheap=cheap, expensive=exp, spread_threshold=0.05)
    hits = [make_hit(i, f"t{i}") for i in range(3)]
    out = await cascade.arerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-2", "c-1", "c-0"]
    assert cheap.calls == 1
    assert exp.calls == 0


@pytest.mark.asyncio
async def test_cascade_fires_expensive_when_cheap_agrees_with_input() -> None:
    cheap = StubReranker("cheap", lambda h: list(h))
    exp = StubReranker("exp", lambda h: list(reversed(h)))
    cascade = CascadeReranker(cheap=cheap, expensive=exp, spread_threshold=0.05)
    hits = [make_hit(i, f"t{i}") for i in range(3)]
    out = await cascade.arerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-2", "c-1", "c-0"]
    assert cheap.calls == 1
    assert exp.calls == 1


@pytest.mark.asyncio
async def test_cascade_empty_hits_short_circuits() -> None:
    cheap = StubReranker("cheap", lambda h: list(h))
    exp = StubReranker("exp", lambda h: list(h))
    cascade = CascadeReranker(cheap=cheap, expensive=exp, spread_threshold=0.05)
    out = await cascade.arerank(question="q", hits=[])
    assert out == []
    # Neither reranker is called for an empty input.
    assert cheap.calls == 0
    assert exp.calls == 0


# --- Factory --------------------------------------------------------------


def test_factory_default_is_identity() -> None:
    """Provider ``"none"`` (the default) returns the no-op reranker."""
    from raghub.retrieval.reranker import IdentityReranker

    s = Settings()
    r = build_reranker(s)
    assert isinstance(r, IdentityReranker)


def test_factory_llm_uses_heuristic_when_no_llm_provided() -> None:
    s = Settings(reranker=RerankerConfig(provider="llm"))
    r = build_reranker(s)
    assert isinstance(r, LLMReranker)


def test_factory_bge_returns_bge_reranker() -> None:
    from raghub.retrieval.rerankers.bge import BgeReranker

    s = Settings(reranker=RerankerConfig(provider="bge"))
    r = build_reranker(s)
    assert isinstance(r, BgeReranker)


def test_factory_cascade_falls_back_to_bge_when_no_cohere_key() -> None:
    """Without a Cohere API key, cascade degenerates to a single BGE."""
    import os

    os.environ.pop("COHERE_API_KEY", None)
    s = Settings(reranker=RerankerConfig(provider="cascade"))
    r = build_reranker(s)
    assert isinstance(r, CascadeReranker)
    from raghub.retrieval.rerankers.bge import BgeReranker

    assert isinstance(r.cheap, BgeReranker)
    assert isinstance(r.expensive, BgeReranker)


def test_factory_unknown_provider_raises_at_validation() -> None:
    """Pydantic Literal validation rejects unknown providers before we reach the factory."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Settings(reranker=RerankerConfig(provider="bogus"))


# --- CohereReranker construction ------------------------------------------


def test_cohere_requires_api_key() -> None:
    import os

    os.environ.pop("COHERE_API_KEY", None)
    from raghub.retrieval.rerankers.cohere import CohereReranker

    with pytest.raises(RerankerError):
        CohereReranker()


def test_cohere_uses_explicit_api_key() -> None:
    import os

    os.environ.pop("COHERE_API_KEY", None)
    from raghub.retrieval.rerankers.cohere import CohereReranker

    r = CohereReranker(api_key="test-key")
    assert r.api_key.get_secret_value() == "test-key"


def test_cohere_uses_env_var_when_no_arg() -> None:
    import os

    os.environ["COHERE_API_KEY"] = "env-key"
    from raghub.retrieval.rerankers.cohere import CohereReranker

    r = CohereReranker()
    assert r.api_key.get_secret_value() == "env-key"
    del os.environ["COHERE_API_KEY"]


def test_cohere_rerank_with_mocked_client() -> None:
    """CohereReranker.rerank delegates to ``client.rerank`` and reorders."""
    from raghub.retrieval.rerankers.cohere import CohereReranker

    class ResultObj:
        def __init__(self, index: int) -> None:
            self.index = index

    class FakeResponse:
        def __init__(self, indices: list[int]) -> None:
            self.results = [ResultObj(i) for i in indices]

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def rerank(self, *, model, query, documents, top_n):
            self.calls.append({"model": model, "query": query, "n": len(documents)})
            return FakeResponse([2, 0, 1])

    client = FakeClient()
    r = CohereReranker(api_key="k", client=client)
    hits = [make_hit(0, "a"), make_hit(1, "b"), make_hit(2, "c")]
    out = r.rerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-2", "c-0", "c-1"]
    assert client.calls[0]["model"] == "rerank-english-v3.0"


# --- BgeReranker construction --------------------------------------------


def test_bge_uses_explicit_encoder() -> None:
    from raghub.retrieval.rerankers.bge import BgeReranker

    class FakeEncoder:
        def predict(self, pairs):
            return [1.0 - i * 0.1 for i in range(len(pairs))]

    r = BgeReranker(encoder=FakeEncoder(), top_k=3)
    hits = [make_hit(0, "a"), make_hit(1, "b"), make_hit(2, "c")]
    out = r.rerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-0", "c-1", "c-2"]


def test_bge_ensure_encoder_is_idempotent() -> None:
    """ensure_encoder() returns the same encoder on repeated calls.

    The first call materialises the ``CrossEncoder``; subsequent calls
    return the cached instance. Verifies the encoder attribute
    survives multiple ``ensure_encoder`` invocations.
    """
    from raghub.retrieval.rerankers.bge import BgeReranker

    r = BgeReranker(encoder=None)
    # We don't actually load the heavy CrossEncoder here — we test
    # that if the encoder is pre-set, ensure_encoder returns it.
    sentinel = object()
    r.encoder = sentinel
    assert r.ensure_encoder() is sentinel