"""Phase 5.1 / 5.3 — ``LongContextRerankPass`` behaviour."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from raghub.config.settings import LongContextConfig
from raghub.models import ChunkRecord, RetrievalHit
from raghub.retrieval.long_context import LongContextRerankPass


class StubLlm:
    """Stand-in LLM that returns ``response`` from ``async_generate``."""

    def __init__(self, model_name: str, response: Any = "") -> None:
        self.model_name = model_name
        self.response = response
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"system": system_prompt, "question": question})
        return self.response


def make_hit(i: int, text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"c-{i}",
        score=1.0,
        chunk=ChunkRecord(
            chunk_id=f"c-{i}",
            document_id="d",
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


# --- eligibility ---------------------------------------------------------


def test_is_eligible_requires_enabled_and_allowlisted_model() -> None:
    cfg = LongContextConfig(enabled=True)
    pass_ = LongContextRerankPass(StubLlm("heuristic-llm"), cfg)
    assert pass_.is_eligible() is False  # not in allowlist

    pass_ok = LongContextRerankPass(StubLlm("claude-3-5-sonnet"), cfg)
    assert pass_ok.is_eligible() is True

    cfg_off = LongContextConfig(enabled=False)
    pass_off = LongContextRerankPass(StubLlm("claude-3-5-sonnet"), cfg_off)
    assert pass_off.is_eligible() is False


def test_is_eligible_no_op_when_llm_has_no_model_name() -> None:
    class Nameless:
        async def async_generate(self, **_):  # pragma: no cover — never called
            return ""

    pass_ = LongContextRerankPass(Nameless(), LongContextConfig(enabled=True))
    assert pass_.is_eligible() is False


# --- rerank --------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_returns_hits_unchanged_when_ineligible() -> None:
    """A disabled pass is a no-op; no LLM call is made."""
    llm = StubLlm("claude-3-5-sonnet", response="SHOULD NOT BE USED")
    pass_ = LongContextRerankPass(llm, LongContextConfig(enabled=False))
    hits = [make_hit(0, "alpha"), make_hit(1, "beta")]
    out = await pass_.rerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-0", "c-1"]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_rerank_empty_hits_short_circuits() -> None:
    pass_ = LongContextRerankPass(
        StubLlm("claude-3-5-sonnet"),
        LongContextConfig(enabled=True),
    )
    out = await pass_.rerank(question="q", hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_rerank_applies_long_context_ordering() -> None:
    payload = (
        '{"items": ['
        '{"chunk_id": "c-2", "score": 0.9, "rationale": "direct"}, '
        '{"chunk_id": "c-0", "score": 0.4, "rationale": "tangential"}'
        ']}'
    )
    llm = StubLlm("claude-3-5-sonnet", response=payload)
    pass_ = LongContextRerankPass(llm, LongContextConfig(enabled=True, candidate_k=3))
    hits = [make_hit(0, "a"), make_hit(1, "b"), make_hit(2, "c")]
    out = await pass_.rerank(question="q", hits=hits)
    # LLM only sees the first 3 candidates (candidate_k=3).
    assert [h.chunk_id for h in out] == ["c-2", "c-0", "c-1"]
    assert llm.calls


@pytest.mark.asyncio
async def test_rerank_falls_back_on_bad_json() -> None:
    """Unparseable JSON keeps the original order."""
    llm = StubLlm("claude-3-5-sonnet", response="not json at all")
    pass_ = LongContextRerankPass(llm, LongContextConfig(enabled=True))
    hits = [make_hit(0, "a"), make_hit(1, "b")]
    out = await pass_.rerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-0", "c-1"]


@pytest.mark.asyncio
async def test_rerank_falls_back_on_schema_violation() -> None:
    """An out-of-range score keeps the original order."""
    payload = (
        '{"items": ['
        '{"chunk_id": "c-1", "score": 2.0, "rationale": "bad score"}'
        ']}'
    )
    llm = StubLlm("claude-3-5-sonnet", response=payload)
    pass_ = LongContextRerankPass(llm, LongContextConfig(enabled=True))
    hits = [make_hit(0, "a"), make_hit(1, "b")]
    out = await pass_.rerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-0", "c-1"]


@pytest.mark.asyncio
async def test_rerank_falls_back_when_llm_raises() -> None:
    """An LLM exception degrades to the original order, never crashes."""

    class Raising:
        model_name = "claude-3-5-sonnet"

        async def async_generate(self, **_):
            raise RuntimeError("upstream down")

    pass_ = LongContextRerankPass(Raising(), LongContextConfig(enabled=True))
    hits = [make_hit(0, "a"), make_hit(1, "b")]
    out = await pass_.rerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-0", "c-1"]


@pytest.mark.asyncio
async def test_rerank_appends_hits_llm_omitted() -> None:
    """When the model drops some candidates they re-appear in the tail."""
    payload = (
        '{"items": ['
        '{"chunk_id": "c-2", "score": 0.9, "rationale": "direct"}'
        ']}'
    )
    llm = StubLlm("claude-3-5-sonnet", response=payload)
    pass_ = LongContextRerankPass(llm, LongContextConfig(enabled=True, candidate_k=10))
    hits = [make_hit(0, "a"), make_hit(1, "b"), make_hit(2, "c")]
    out = await pass_.rerank(question="q", hits=hits)
    assert [h.chunk_id for h in out] == ["c-2", "c-0", "c-1"]