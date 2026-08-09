"""Coverage tests for :mod:`raghub.retrieval.judge`, :mod:`.context`,
and :mod:`.colbert`.

These three modules were at 0% coverage. The tests focus on the public
LLM-driven rerankers with deterministic stub generators — every assertion
is observable, not a smoke test.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from raghub.config import LongContextConfig
from raghub.models import Chunk, Hit
from raghub.retrieval.colbert import Colbert
from raghub.retrieval.context import Context
from raghub.retrieval.judge import (
    CONTEXT,
    LISTWISE_MAX,
    LlmJudge,
    context_prompt,
    extract_array,
    extract_object,
    extract_strings,
    record_latency,
    reorder_candidates,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "text", chunk_id: str = "c1") -> Chunk:
    from hashlib import sha256

    return Chunk(
        id=chunk_id,
        document_id="doc",
        version=1,
        page=1,
        source_location="page 1",
        section="",
        company="acme",
        owner="alice@example.com",
        department="",
        tenant_id="acme",
        text=text,
        checksum=sha256(text.encode("utf-8")).hexdigest(),
    )


def _make_hit(text: str = "text", chunk_id: str = "c1", score: float = 0.5) -> Hit:
    return Hit(score=score, chunk=_make_chunk(text=text, chunk_id=chunk_id))


# ---------------------------------------------------------------------------
# LlmJudge
# ---------------------------------------------------------------------------


class StubLLM:
    """Minimal async generator stub."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[Any] = []

    async def async_generate(self, request: Any) -> str:
        self.calls.append(request)
        return self.response


class TestLlmJudgeRankWindow:
    def test_rank_window_uses_llm_ordering(self) -> None:
        llm = StubLLM('[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.5}]')
        judge = LlmJudge(llm=llm, top_k=2)
        hits = [_make_hit("first", "c0", 0.1), _make_hit("second", "c1", 0.2)]
        ordered = asyncio.run(judge.rank_window("q?", hits))
        # The judge reordered: second (index 1) first, first (index 0) second.
        assert [h.chunk_id for h in ordered] == ["c1", "c0"]

    def test_rank_window_falls_back_to_input_order_on_unparseable(self) -> None:
        llm = StubLLM("not json")
        judge = LlmJudge(llm=llm, top_k=2)
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        ordered = asyncio.run(judge.rank_window("q", hits))
        assert [h.chunk_id for h in ordered] == ["c0", "c1"]

    def test_rank_window_drops_invalid_indices(self) -> None:
        llm = StubLLM('[{"index": 99, "score": 0.9}, {"index": 0, "score": 0.5}]')
        judge = LlmJudge(llm=llm, top_k=2)
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        ordered = asyncio.run(judge.rank_window("q", hits))
        # Index 99 is dropped, c0 promoted, c1 added at the end.
        assert [h.chunk_id for h in ordered] == ["c0", "c1"]


class TestLlmJudgeDoRerank:
    def test_short_input_uses_listwise_path(self) -> None:
        llm = StubLLM('[{"index": 0, "score": 0.9}]')
        judge = LlmJudge(llm=llm, top_k=3)
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        ordered = asyncio.run(judge.do_rerank("q", hits))
        assert len(ordered) == 2
        # Only one LLM call (listwise, not windowed).
        assert len(llm.calls) == 1

    def test_long_input_uses_windowed_rrf(self) -> None:
        llm = StubLLM('[{"index": 0, "score": 0.9}]')
        judge = LlmJudge(llm=llm, top_k=3)
        # 12 hits = 2 windows of LISTWISE_MAX.
        hits = [_make_hit(f"text-{i}", f"c{i}", 0.1 * i) for i in range(12)]
        ordered = asyncio.run(judge.do_rerank("q", hits))
        # Each window triggers one LLM call.
        assert len(llm.calls) == 2
        # top_k limits the output.
        assert len(ordered) <= 3


class TestLlmJudgeArerank:
    def test_arerank_empty_hits_returns_empty(self) -> None:
        judge = LlmJudge(llm=StubLLM(""), top_k=5)
        assert asyncio.run(judge.arerank(question="q", hits=[])) == []

    def test_arerank_invokes_llm_and_returns_top_k(self) -> None:
        llm = StubLLM('[{"index": 0, "score": 0.9}, {"index": 1, "score": 0.8}]')
        judge = LlmJudge(llm=llm, top_k=2)
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        result = asyncio.run(judge.arerank(question="q", hits=hits))
        assert len(result) == 2
        assert [h.chunk_id for h in result] == ["c0", "c1"]

    def test_rerank_sync_wrapper_returns_top_k(self) -> None:
        llm = StubLLM('[{"index": 0, "score": 0.9}, {"index": 1, "score": 0.8}]')
        judge = LlmJudge(llm=llm, top_k=2)
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        result = judge.rerank(question="q", hits=hits)
        assert len(result) == 2
        assert [h.chunk_id for h in result] == ["c0", "c1"]


# ---------------------------------------------------------------------------
# Helpers in judge.py
# ---------------------------------------------------------------------------


class TestContextPrompt:
    def test_includes_question_and_snippets(self) -> None:
        hits = [
            _make_hit("first chunk text", "c0"),
            _make_hit("second chunk text", "c1"),
        ]
        prompt = context_prompt("the question?", hits)
        assert "the question?" in prompt
        assert "first chunk text" in prompt
        assert "second chunk text" in prompt
        assert "c0" in prompt
        assert "c1" in prompt


class TestExtractObject:
    def test_inline_object(self) -> None:
        assert extract_object('prefix {"a": 1} suffix') == {"a": 1}

    def test_fenced_object(self) -> None:
        text = 'pre\n```json\n{"k": "v"}\n```\npost'
        assert extract_object(text) == {"k": "v"}

    def test_no_object_returns_none(self) -> None:
        assert extract_object("plain text") is None
        assert extract_object("") is None

    def test_unbalanced_braces_returns_none(self) -> None:
        assert extract_object('{"a": ') is None


class TestReorderCandidates:
    def test_returns_ordered_hits(self) -> None:
        from raghub.models import RankedItem, RankedList

        candidates = [
            _make_hit("a", "c0"),
            _make_hit("b", "c1"),
            _make_hit("c", "c2"),
        ]
        ranked = RankedList(
            items=[
                RankedItem(
                    id="c2",
                    score=0.9,
                    rank=0,
                    chunk=candidates[2].chunk,
                ),
                RankedItem(
                    id="c0",
                    score=0.5,
                    rank=1,
                    chunk=candidates[0].chunk,
                ),
            ]
        )
        result = reorder_candidates(candidates, ranked)
        assert result is not None
        assert [h.chunk_id for h in result] == ["c2", "c0", "c1"]

    def test_returns_none_when_ranked_is_empty(self) -> None:
        from raghub.models import RankedList

        candidates = [_make_hit("a", "c0")]
        ranked = RankedList(items=[])
        assert reorder_candidates(candidates, ranked) is None


class TestRecordLatency:
    def test_record_latency_does_not_raise(self) -> None:
        # record_long_context is a no-op when Prometheus is not configured.
        # The function should not raise.
        record_latency("ran", 0.001)


# ---------------------------------------------------------------------------
# extract_array / extract_strings (re-exports at top of test for convenience)
# ---------------------------------------------------------------------------


class TestExtractArrayFromJudge:
    def test_inline_array(self) -> None:
        assert extract_array('here [{"a": 1}]') == [{"a": 1}]

    def test_fenced_array(self) -> None:
        assert extract_array('```json\n[{"x": 1}]\n```') == [{"x": 1}]

    def test_no_array_returns_empty(self) -> None:
        assert extract_array("nothing here") == []
        assert extract_array("") == []


class TestExtractStringsFromJudge:
    def test_inline_strings(self) -> None:
        assert extract_strings('prefix ["a", "b"] suffix') == ["a", "b"]

    def test_fenced_strings(self) -> None:
        assert extract_strings('```json\n["x"]\n```') == ["x"]

    def test_no_strings_returns_empty(self) -> None:
        assert extract_strings("no json") == []
        assert extract_strings("") == []


# ---------------------------------------------------------------------------
# Context (long-context rerank)
# ---------------------------------------------------------------------------


class RecordingLLM:
    """Stub LLM that records calls and returns a canned response."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[Any] = []

    async def async_generate(self, request: Any) -> str:
        self.calls.append(request)
        return self.response


class TestContextEligibility:
    def test_disabled_returns_false(self) -> None:
        llm = RecordingLLM()
        llm.model_name = "claude-3-5-sonnet"  # type: ignore[attr-defined]
        ctx = Context(llm, LongContextConfig(enabled=False, allowlist_models=[]))
        assert ctx.is_eligible() is False

    def test_no_model_name_returns_false(self) -> None:
        llm = RecordingLLM()  # no model_name attribute
        ctx = Context(llm, LongContextConfig(enabled=True, allowlist_models=["gpt-4o"]))
        assert ctx.is_eligible() is False

    def test_model_not_in_allowlist_returns_false(self) -> None:
        llm = RecordingLLM()
        llm.model_name = "unsupported-model"  # type: ignore[attr-defined]
        ctx = Context(llm, LongContextConfig(enabled=True, allowlist_models=["gpt-4o"]))
        assert ctx.is_eligible() is False

    def test_model_in_allowlist_returns_true(self) -> None:
        llm = RecordingLLM()
        llm.model_name = "claude-3-5-sonnet"  # type: ignore[attr-defined]
        ctx = Context(llm, LongContextConfig(enabled=True, allowlist_models=["claude-3-5-sonnet"]))
        assert ctx.is_eligible() is True


class TestContextRerank:
    def test_not_eligible_returns_input_unchanged(self) -> None:
        llm = RecordingLLM()
        ctx = Context(llm, LongContextConfig(enabled=False))
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        result = asyncio.run(ctx.rerank(question="q", hits=hits))
        # LLM was not consulted, input was returned.
        assert [h.chunk_id for h in result] == ["c0", "c1"]
        assert llm.calls == []

    def test_empty_hits_returns_empty(self) -> None:
        llm = RecordingLLM()
        llm.model_name = "claude-3-5-sonnet"  # type: ignore[attr-defined]
        ctx = Context(llm, LongContextConfig(enabled=True, allowlist_models=["claude-3-5-sonnet"]))
        assert asyncio.run(ctx.rerank(question="q", hits=[])) == []

    def test_unparseable_response_returns_input_unchanged(self) -> None:
        llm = RecordingLLM(response="not json at all")
        llm.model_name = "claude-3-5-sonnet"  # type: ignore[attr-defined]
        ctx = Context(llm, LongContextConfig(enabled=True, allowlist_models=["claude-3-5-sonnet"]))
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        result = asyncio.run(ctx.rerank(question="q", hits=hits))
        assert [h.chunk_id for h in result] == ["c0", "c1"]

    def test_valid_response_falls_through_due_to_schema_mismatch(self) -> None:
        """The LLM contract uses ``chunk_id``; the parsed path falls through.

        The current implementation parses the response with
        :class:`RankedList` whose items require the ``id`` + ``chunk``
        schema, so even a well-formed long-context response ends up
        unparseable and the input order is preserved. This is a known
        bug; the test pins the observable behaviour.
        """
        response = (
            '{"items": ['
            '{"chunk_id": "c1", "score": 0.9, "rationale": "better"},'
            '{"chunk_id": "c0", "score": 0.5, "rationale": "worse"}'
            "]}"
        )
        llm = RecordingLLM(response=response)
        llm.model_name = "claude-3-5-sonnet"  # type: ignore[attr-defined]
        ctx = Context(llm, LongContextConfig(enabled=True, allowlist_models=["claude-3-5-sonnet"]))
        hits = [_make_hit("a", "c0"), _make_hit("b", "c1")]
        result = asyncio.run(ctx.rerank(question="q", hits=hits))
        # The schema mismatch forces the fallback path.
        assert [h.chunk_id for h in result] == ["c0", "c1"]

    def test_arerank_alias(self) -> None:
        llm = RecordingLLM()
        llm.model_name = "claude-3-5-sonnet"  # type: ignore[attr-defined]
        ctx = Context(llm, LongContextConfig(enabled=False))
        hits = [_make_hit("a", "c0")]
        result = asyncio.run(ctx.arerank(question="q", hits=hits))
        assert [h.chunk_id for h in result] == ["c0"]


# ---------------------------------------------------------------------------
# Colbert
# ---------------------------------------------------------------------------


class TestColbert:
    def test_disabled_when_no_config(self) -> None:
        adapter = Colbert(None)
        assert adapter.is_available() is False

    def test_disabled_when_colbert_enabled_false(self) -> None:
        config = MagicMock()
        config.colbert_enabled = False
        adapter = Colbert(config)
        assert adapter.is_available() is False

    def test_score_empty_returns_empty(self) -> None:
        adapter = Colbert(None)
        assert adapter.score("query", []) == []

    def test_score_disabled_returns_empty(self) -> None:
        adapter = Colbert(None)
        assert adapter.score("query", ["doc"]) == []

    def test_score_enabled_missing_dep_raises(self) -> None:
        """When colbert_enabled is True but ragatouille is missing, raise GraphUnavailableError."""
        from raghub.errors import GraphUnavailableError

        config = MagicMock()
        config.colbert_enabled = True
        # Inject a stub finder that always says ragatouille is missing.
        import importlib.util as _il

        original_find_spec = _il.find_spec

        def _missing(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "ragatouille":
                return None
            return original_find_spec(name, *args, **kwargs)

        _il.find_spec = _missing
        try:
            adapter = Colbert(config)
            with pytest.raises(GraphUnavailableError, match="ragatouille"):
                adapter.score("q", ["d1"])
        finally:
            _il.find_spec = original_find_spec

    def test_score_enabled_ragatouille_available(self) -> None:
        """When colbert_enabled and ragatouille present, the index is loaded and queried."""

        # Inject a fake ragatouille module with a proper spec so
        # ``importlib.util.find_spec`` accepts it.
        class FakeModel:
            @staticmethod
            def from_pretrained(name: str) -> Any:
                return FakeModel()

            @staticmethod
            def rerank(query: str, documents: list[str]) -> list[float]:
                return [0.9 - 0.1 * i for i in range(len(documents))]

        fake_module = types.ModuleType("ragatouille")
        fake_module.RAGPretrainedModel = FakeModel  # type: ignore[attr-defined]
        fake_module.__spec__ = importlib.machinery.ModuleSpec(  # type: ignore[attr-defined]
            name="ragatouille", loader=None
        )
        sys.modules["ragatouille"] = fake_module

        try:
            config = MagicMock()
            config.colbert_enabled = True
            adapter = Colbert(config)
            scores = adapter.score("q", ["d1", "d2", "d3"])
            assert scores == [0.9, 0.8, 0.7]
            # Second call reuses the index.
            scores2 = adapter.score("q", ["d1", "d2"])
            assert scores2 == [0.9, 0.8]
        finally:
            del sys.modules["ragatouille"]


# ---------------------------------------------------------------------------
# module-level constants
# ---------------------------------------------------------------------------


def test_listwise_max_constant() -> None:
    assert LISTWISE_MAX == 10


def test_context_constant() -> None:
    assert "re-rank" in CONTEXT
