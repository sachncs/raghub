"""Tests for the Judge evaluator and the parse helper."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from raghub.errors import ConfigurationError
from raghub.eval import Gate, Judge, compare, parse
from raghub.llm import GenerationRequest

# ---------------------------------------------------------------------------
# Pure parser tests (parse)
# ---------------------------------------------------------------------------


def test_parse_plain_decimal() -> None:
    """A bare ``0.85`` parses to 0.85."""
    assert parse("0.85") == 0.85


def test_parse_with_surrounding_text() -> None:
    """A number embedded in prose is extracted."""
    assert parse("The score is 0.7") == 0.7


def test_parse_clamps_above_one() -> None:
    """Values above 1.0 are clamped to 1.0."""
    assert parse("1.5") == 1.0


def test_parse_clamps_negative_to_zero() -> None:
    """Negative values are clamped to 0.0."""
    assert parse("-0.5") == 0.0
    assert parse("-1.0") == 0.0


def test_parse_returns_none_when_no_number() -> None:
    """A response without a 0..1 float returns None."""
    assert parse("I cannot score this") is None
    assert parse("") is None
    assert parse("today is 2024") is None


def test_parse_ignores_year_like_numbers() -> None:
    """Years like 2024 are out of range and not picked up."""
    assert parse("the year 2024") is None


def test_parse_integer_zero_and_one() -> None:
    """Plain ``0`` and ``1`` parse correctly."""
    assert parse("0") == 0.0
    assert parse("1") == 1.0
    assert parse("0.0") == 0.0
    assert parse("1.0") == 1.0


def test_parse_negative_one_clamped() -> None:
    """``-1.0`` clamps to 0.0."""
    assert parse("-1.0") == 0.0


# ---------------------------------------------------------------------------
# Judge with a fake generator
# ---------------------------------------------------------------------------


class FakeGenerator:
    """Stub LLM that returns a fixed response and counts calls."""

    def __init__(self, responses: list[str] | str, raise_on: set[int] | None = None) -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self._raise_on = raise_on or set()
        self.calls: list[GenerationRequest] = []

    async def async_generate(self, request) -> str:
        self.calls.append(request)
        idx = len(self.calls) - 1
        if idx in self._raise_on:
            raise RuntimeError(f"simulated failure on call {idx}")
        return self._responses[min(idx, len(self._responses) - 1)]


def run_async(coro):
    """Synchronously run an async coroutine to completion."""
    return asyncio.run(coro)


def test_llm_judge_faithfulness_returns_float_in_range() -> None:
    """A faithful response gives a score in [0.0, 1.0]."""
    fake = FakeGenerator("0.85")
    judge = Judge(fake)
    score = run_async(judge.faithfulness("The answer is 42.", ["context"]))
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.85)


def test_llm_judge_relevance_returns_float_in_range() -> None:
    """A relevance response gives a score in [0.0, 1.0]."""
    fake = FakeGenerator("0.6")
    judge = Judge(fake)
    score = run_async(judge.answer_relevance("An answer", "A question"))
    assert score == pytest.approx(0.6)


def test_llm_judge_returns_zero_when_no_number_in_response() -> None:
    """A response with no parsable number raises (not a silent 0.0)."""
    from raghub.errors import EvaluationError

    fake = FakeGenerator("I cannot score this")
    judge = Judge(fake)
    with pytest.raises(EvaluationError):
        run_async(judge.faithfulness("anything", ["anything"]))


def test_llm_judge_clamps_overflow_to_one() -> None:
    """A response of 1.5 is clamped to 1.0."""
    fake = FakeGenerator("1.5")
    judge = Judge(fake)
    score = run_async(judge.faithfulness("a", ["a"]))
    assert score == 1.0


def test_llm_judge_uses_provided_generator() -> None:
    """The supplied generator is the one called."""
    fake = FakeGenerator("0.5")
    judge = Judge(fake)
    run_async(judge.faithfulness("a", ["a"]))
    assert len(fake.calls) == 1


def test_llm_judge_prompt_includes_answer_and_contexts() -> None:
    """The faithfulness prompt contains both the answer and the contexts."""
    fake = FakeGenerator("0.7")
    judge = Judge(fake)
    run_async(judge.faithfulness("the answer is 42", ["first context", "second context"]))
    assert len(fake.calls) == 1
    prompt = fake.calls[0].system_prompt
    assert "the answer is 42" in prompt
    assert "first context" in prompt
    assert "second context" in prompt


def test_llm_judge_relevance_prompt_includes_question() -> None:
    """The relevance prompt contains the question."""
    fake = FakeGenerator("0.5")
    judge = Judge(fake)
    run_async(judge.answer_relevance("the answer", "the question"))
    prompt = fake.calls[0].system_prompt
    assert "the question" in prompt
    assert "the answer" in prompt


def test_llm_judge_retries_on_failed_parse() -> None:
    """First response unparseable, second parsed → score is the second."""
    fake = FakeGenerator(["oops", "0.7"])
    judge = Judge(fake, max_retries=1)
    score = run_async(judge.faithfulness("a", ["a"]))
    assert score == 0.7
    assert len(fake.calls) == 2


def test_llm_judge_retries_on_generator_exception() -> None:
    """Generator raising on first call → retry uses the second response."""
    fake = FakeGenerator(["0.4"], raise_on={0})
    judge = Judge(fake, max_retries=1)
    score = run_async(judge.faithfulness("a", ["a"]))
    assert score == 0.4
    assert len(fake.calls) == 2


def test_llm_judge_gives_up_after_max_retries() -> None:
    """All retries fail to parse → raises EvaluationError (not a silent 0.0)."""
    from raghub.errors import EvaluationError

    fake = FakeGenerator(["nope", "still nope", "really nope"])
    judge = Judge(fake, max_retries=2)
    with pytest.raises(EvaluationError):
        run_async(judge.faithfulness("a", ["a"]))
    assert len(fake.calls) == 3


def test_llm_judge_no_retry_when_first_succeeds() -> None:
    """A single attempt (max_retries=0) does not retry on success."""
    fake = FakeGenerator("0.9")
    judge = Judge(fake, max_retries=0)
    score = run_async(judge.faithfulness("a", ["a"]))
    assert score == 0.9
    assert len(fake.calls) == 1


def test_llm_judge_default_max_retries_is_one() -> None:
    """Default ``max_retries=1`` allows 2 attempts total."""
    fake = FakeGenerator(["nope", "0.3"])
    judge = Judge(fake)
    score = run_async(judge.faithfulness("a", ["a"]))
    assert score == 0.3
    assert len(fake.calls) == 2


def test_llm_judge_self_loop_does_not_exist() -> None:
    """Judge does not create its own event loop; it must be called from one.

    When called outside a loop, async_generate never runs — the
    caller is expected to be in a coroutine. This is a contract
    test: the Judge class itself doesn't expose any sync API.
    """
    fake = FakeGenerator("0.5")
    judge = Judge(fake)
    # Judge should not have a sync ``faithfulness`` or
    # ``answer_relevance`` method.
    assert not hasattr(judge, "faithfulness_sync")
    assert not hasattr(judge, "answer_relevance_sync")


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


def test_quality_gate_constructor_with_thresholds() -> None:
    """The constructor accepts a metric-name → threshold mapping."""
    gate = Gate({"recall_at_5": 0.7, "faithfulness": 0.8})
    assert gate.thresholds["recall_at_5"] == (0.7, "min")
    assert gate.thresholds["faithfulness"] == (0.8, "min")


def test_quality_gate_passes_when_all_above_threshold() -> None:
    """A metric above its threshold passes silently."""
    gate = Gate({"recall_at_5": 0.7})
    gate.check({"recall_at_5": 0.9})


def test_quality_gate_raises_when_below_threshold() -> None:
    """A metric below its threshold raises ConfigurationError."""
    gate = Gate({"recall_at_5": 0.7})
    with pytest.raises(ConfigurationError, match="recall_at_5"):
        gate.check({"recall_at_5": 0.5})


def test_quality_gate_error_message_includes_value_and_threshold() -> None:
    """The error message includes the actual value and the threshold."""
    gate = Gate({"recall_at_5": 0.7})
    with pytest.raises(ConfigurationError) as exc_info:
        gate.check({"recall_at_5": 0.5})
    assert "0.500" in str(exc_info.value)
    assert "0.7" in str(exc_info.value)


def test_quality_gate_max_mode_passes_when_below_threshold() -> None:
    """In max mode, the metric must be <= threshold."""
    gate = Gate({"latency_ms": 200}, default_mode="max")
    gate.check({"latency_ms": 150})


def test_quality_gate_max_mode_raises_when_above_threshold() -> None:
    """In max mode, a metric above the threshold raises."""
    gate = Gate({"latency_ms": 200}, default_mode="max")
    with pytest.raises(ConfigurationError, match="latency_ms"):
        gate.check({"latency_ms": 250})


def test_quality_gate_per_metric_mode_override() -> None:
    """A per-metric mode override beats the default."""
    gate = Gate({"high": 0.5, "low": 100.0}, default_mode="min")
    gate.add("high", 0.5, mode="min")  # explicit
    gate.add("low", 100.0, mode="max")  # explicit override
    gate.check({"high": 0.6, "low": 50.0})


def test_quality_gate_fluent_builder() -> None:
    """``add()`` returns self for chaining."""
    gate = Gate().add("recall_at_5", 0.7).add("faithfulness", 0.8)
    assert gate.thresholds["recall_at_5"] == (0.7, "min")
    assert gate.thresholds["faithfulness"] == (0.8, "min")


def test_quality_gate_fluent_builder_with_max_mode() -> None:
    """The fluent builder accepts a per-metric mode override."""
    gate = Gate().add("recall_at_5", 0.7).add("latency_ms", 200, mode="max")
    gate.check({"recall_at_5": 0.9, "latency_ms": 150})


def test_quality_gate_missing_metric_raises() -> None:
    """A threshold for a metric not in the input dict raises."""
    gate = Gate({"recall_at_5": 0.7})
    with pytest.raises(ConfigurationError, match="missing"):
        gate.check({})


def test_quality_gate_equality_at_threshold_passes() -> None:
    """A metric exactly at its threshold passes (min mode)."""
    gate = Gate({"recall_at_5": 0.7})
    gate.check({"recall_at_5": 0.7})  # exactly at threshold; not strictly less


def test_quality_gate_equality_at_threshold_passes_max() -> None:
    """A metric exactly at its threshold passes (max mode)."""
    gate = Gate({"latency_ms": 200}, default_mode="max")
    gate.check({"latency_ms": 200})


def test_quality_gate_invalid_default_mode_raises() -> None:
    """Constructing with an invalid default_mode raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="default_mode"):
        Gate({"x": 0.5}, default_mode="invalid")


def test_quality_gate_invalid_add_mode_raises() -> None:
    """Adding a threshold with an invalid mode raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="mode"):
        Gate().add("y", 0.5, mode="invalid")


def test_quality_gate_report_returns_pass_status() -> None:
    """``report()`` returns a (value, threshold, passed, mode) tuple per metric."""
    gate = Gate({"recall_at_5": 0.7})
    report = gate.report({"recall_at_5": 0.9, "extra": 1.0})
    assert report["recall_at_5"] == (0.9, 0.7, True, "min")


def test_quality_gate_report_marks_fail_for_below_threshold() -> None:
    """``report()`` marks the metric as failed when below the threshold."""
    gate = Gate({"recall_at_5": 0.7})
    report = gate.report({"recall_at_5": 0.5})
    assert report["recall_at_5"] == (0.5, 0.7, False, "min")


def test_quality_gate_report_marks_fail_for_missing() -> None:
    """``report()`` marks a missing metric as failed (value is None)."""
    gate = Gate({"recall_at_5": 0.7})
    report = gate.report({})
    assert report["recall_at_5"] == (None, 0.7, False, "min")


def test_quality_gate_report_max_mode() -> None:
    """``report()`` evaluates max mode correctly."""
    gate = Gate({"latency_ms": 200}, default_mode="max")
    assert gate.report({"latency_ms": 150})["latency_ms"] == (150.0, 200.0, True, "max")
    assert gate.report({"latency_ms": 250})["latency_ms"] == (250.0, 200.0, False, "max")


def test_quality_gate_multiple_breaches_reported_together() -> None:
    """When multiple metrics breach, the error names all of them."""
    gate = Gate({"recall_at_5": 0.7, "faithfulness": 0.8})
    with pytest.raises(ConfigurationError) as exc_info:
        gate.check({"recall_at_5": 0.5, "faithfulness": 0.5})
    msg = str(exc_info.value)
    assert "recall_at_5" in msg
    assert "faithfulness" in msg


# ---------------------------------------------------------------------------
# compare harness tests
# ---------------------------------------------------------------------------


class FakeEvaluator:
    """Mimics the Evaluator protocol without any dataset loading."""

    benchmark: str = "fake"

    async def evaluate(self, examples, *, response_factory):
        results = []
        from raghub.eval import Metrics
        from raghub.models import Result

        for example in examples:
            out = await response_factory(example)
            if isinstance(out, tuple) and len(out) == 4:
                answer, contexts, retrieved_ids, relevant_ids = out
            elif isinstance(out, dict):
                answer = out.get("answer", "")
                contexts = out.get("contexts", []) or []
                retrieved_ids = out.get("retrieved_ids", []) or []
                relevant_ids = out.get("relevant_ids", []) or list(example.get("relevant_ids", []))
            else:
                answer = out
                retrieved_ids = []
                relevant_ids = list(example.get("relevant_ids", []))
                contexts = []
            metrics = Metrics.evaluate(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                answer=answer,
                contexts=contexts,
                question=example.get("question", ""),
                ground_truth=str(example.get("answer", "")),
                k=5,
            )
            results.append(
                Result(
                    benchmark=self.benchmark,
                    example_id=str(example.get("id", "")),
                    metrics=metrics,
                    passed=True,
                    details={"predicted": answer},
                )
            )
        return results


class FakeRAG:
    """Minimal RAG stub whose ``aquery`` returns the answer string.

    Mirrors the real ``Response`` path: :func:`compare` reads
    ``response.answer`` off the query result and hands the string to
    the evaluator.
    """

    def __init__(self, answer: str = "Paris", contexts: list[str] | None = None) -> None:
        self.answer = answer
        self.contexts = contexts or ["Capital of France is Paris."]
        self.aquery_call_log: list[str] = []

    async def aquery(self, question: str) -> Any:
        """Record the question and return a Response-shaped answer."""
        self.aquery_call_log.append(question)
        return SimpleNamespace(answer=self.answer)


def test_compare_runs_both_rags_and_reports_diffs() -> None:
    """compare invokes both A and B and returns per-metric diffs."""
    rag_a = FakeRAG(answer="Paris")
    rag_b = FakeRAG(answer="Paris")
    examples = [
        {"question": "q1", "answer": "Paris", "contexts": ["the capital"]},
        {"question": "q2", "answer": "Paris", "contexts": ["the capital"]},
    ]

    async def runner() -> dict:
        return await compare(
            rag_a=rag_a,
            rag_b=rag_b,
            examples=examples,
            evaluator=FakeEvaluator(),
        )

    result = asyncio.run(runner())
    assert "a_metrics" in result
    assert "b_metrics" in result
    assert "metric_diffs" in result
    assert "winner" in result
    assert "gate_passed" in result
    assert result["gate_passed"] is True
    assert result["winner"] == "tie"


def test_compare_winner_b_when_better() -> None:
    """When B's answer is closer to the gold, winner is 'b'."""
    rag_a = FakeRAG(answer="London")
    rag_b = FakeRAG(answer="Paris")
    examples = [
        {"question": "q1", "answer": "Paris", "contexts": ["the capital"]},
    ]

    result = asyncio.run(
        compare(
            rag_a=rag_a,
            rag_b=rag_b,
            examples=examples,
            evaluator=FakeEvaluator(),
        )
    )
    assert result["winner"] == "b"


def test_compare_winner_a_when_a_better() -> None:
    """When A's answer is closer to the gold, winner is 'a'."""
    rag_a = FakeRAG(answer="Paris")
    rag_b = FakeRAG(answer="London")
    examples = [
        {"question": "q1", "answer": "Paris", "contexts": ["the capital"]},
    ]

    result = asyncio.run(
        compare(
            rag_a=rag_a,
            rag_b=rag_b,
            examples=examples,
            evaluator=FakeEvaluator(),
        )
    )
    assert result["winner"] == "a"


def test_compare_empty_examples_returns_tie() -> None:
    """Empty examples produce no metrics; the winner is 'tie'."""
    rag_a = FakeRAG()
    rag_b = FakeRAG()
    result = asyncio.run(
        compare(
            rag_a=rag_a,
            rag_b=rag_b,
            examples=[],
            evaluator=FakeEvaluator(),
        )
    )
    assert result["winner"] == "tie"
    assert result["gate_passed"] is True


def test_compare_gate_passed_when_metrics_above_threshold() -> None:
    """A gate with a low threshold passes both RAGs."""
    rag_a = FakeRAG()
    rag_b = FakeRAG()
    gate = Gate({"recall_at_5": 0.0})  # trivially passes
    result = asyncio.run(
        compare(
            rag_a=rag_a,
            rag_b=rag_b,
            examples=[{"question": "q", "answer": "a", "contexts": ["c"]}],
            evaluator=FakeEvaluator(),
            gate=gate,
        )
    )
    assert result["gate_passed"] is True


def test_compare_gate_raises_when_a_below_threshold() -> None:
    """A gate raises ConfigurationError when A's metrics breach it."""
    from raghub.errors import ConfigurationError

    rag_a = FakeRAG()
    rag_b = FakeRAG()
    gate = Gate({"recall_at_5": 999.0})  # impossible threshold
    with pytest.raises(ConfigurationError, match="Gate failed"):
        asyncio.run(
            compare(
                rag_a=rag_a,
                rag_b=rag_b,
                examples=[{"question": "q", "answer": "a", "contexts": ["c"]}],
                evaluator=FakeEvaluator(),
                gate=gate,
            )
        )


def test_compare_calls_each_rag_once_per_example() -> None:
    """Each RAG is queried exactly once per example."""
    rag_a = FakeRAG()
    rag_b = FakeRAG()
    examples = [
        {"question": "q1", "answer": "a", "contexts": ["c"]},
        {"question": "q2", "answer": "a", "contexts": ["c"]},
        {"question": "q3", "answer": "a", "contexts": ["c"]},
    ]
    asyncio.run(
        compare(
            rag_a=rag_a,
            rag_b=rag_b,
            examples=examples,
            evaluator=FakeEvaluator(),
        )
    )
    assert len(rag_a.aquery_call_log) == 3
    assert len(rag_b.aquery_call_log) == 3


# ---------------------------------------------------------------------------
# CLI entry points: raghub.evaluation.financebench / frames
# ---------------------------------------------------------------------------


def test_evaluation_module_exports() -> None:
    """raghub.evaluation exports financebench, frames."""
    import raghub.evaluation as ev_module

    assert ev_module.__all__ == ["financebench", "frames"]


def test_evaluation_app_is_typer() -> None:
    """The eval sub-app is a Typer."""
    from typer import Typer

    import raghub.evaluation as ev_module

    assert isinstance(ev_module.app, Typer)


def test_financebench_cli_empty_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub eval financebench --examples 0`` writes an empty summary."""
    from typer.testing import CliRunner

    import raghub.evaluation as ev_module
    from raghub.eval import Finance

    class EmptyFinance(Finance):
        def ensure_examples(self) -> list[dict[str, Any]]:
            return []

    runner = CliRunner()
    monkeypatch.setattr(ev_module, "Finance", EmptyFinance)
    result = runner.invoke(ev_module.app, ["financebench", "--examples", "0"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["benchmark"] == "financebench"
    assert payload["summary"]["count"] == 0
    assert payload["summary"]["pass_rate"] == 0.0
    assert payload["results"] == []


def test_financebench_cli_with_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub eval financebench --examples 3`` scores three examples."""
    from typer.testing import CliRunner

    import raghub.evaluation as ev_module
    from raghub.eval import Finance

    class StubFinance(Finance):
        def ensure_examples(self) -> list[dict[str, Any]]:
            return [
                {"question": "q1", "answer": "a1", "contexts": ["c1"]},
                {"question": "q2", "answer": "a2", "contexts": ["c2"]},
                {"question": "q3", "answer": "a3", "contexts": ["c3"]},
            ]

    runner = CliRunner()
    monkeypatch.setattr(ev_module, "Finance", StubFinance)
    result = runner.invoke(ev_module.app, ["financebench", "--examples", "3"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["count"] == 3


def test_frames_cli_with_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub eval frames`` with positive examples scores them."""
    from typer.testing import CliRunner

    import raghub.evaluation as ev_module
    from raghub.eval import Frames

    class StubFrames(Frames):
        def ensure_examples(self) -> list[dict[str, Any]]:
            return [
                {"question": "q1", "answer": "a1", "contexts": ["c1"]},
                {"question": "q2", "answer": "a2", "contexts": ["c2"]},
            ]

    runner = CliRunner()
    monkeypatch.setattr(ev_module, "Frames", StubFrames)
    result = runner.invoke(ev_module.app, ["frames", "--examples", "2"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["benchmark"] == "frames"
    assert payload["summary"]["count"] == 2


def test_frames_cli_zero_examples_means_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub eval frames --examples 0`` includes every returned example."""
    from typer.testing import CliRunner

    import raghub.evaluation as ev_module
    from raghub.eval import Frames

    examples = [{"question": f"q{i}", "answer": "a", "contexts": ["c"]} for i in range(5)]

    class StubFrames(Frames):
        def ensure_examples(self) -> list[dict[str, Any]]:
            return examples

    runner = CliRunner()
    monkeypatch.setattr(ev_module, "Frames", StubFrames)
    result = runner.invoke(ev_module.app, ["frames", "--examples", "0"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["count"] == 5
