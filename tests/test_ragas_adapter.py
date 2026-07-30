"""Tests for the RagasAdapter.

The adapter wraps the optional ``[ragas]`` extra. Most tests
exercise the data-translation helpers (``_build_dataset``,
``_extract_scores``) which don't require ragas to be installed.
The full evaluate() path is gated behind a pytest.importorskip
on ragas.
"""

from __future__ import annotations

from typing import Any

import pytest

from raghub.errors import ConfigurationError, MissingDep

# ---------------------------------------------------------------------------
# Translation helpers (no ragas dependency)
# ---------------------------------------------------------------------------


class _FakeRagasResult:
    """Stand-in for a ragas EvaluationResult."""

    def __init__(self, scores: dict[str, list[float]]):
        self.scores = scores


class _FakeRagasModule:
    """Stub ragas module that records what was passed to evaluate()."""

    def __init__(self, raise_on_evaluate: Exception | None = None) -> None:
        self.raise_on_evaluate = raise_on_evaluate
        self.last_kwargs: dict[str, Any] | None = None
        self.last_dataset: Any = None

    def evaluate(self, dataset: Any, **kwargs: Any) -> _FakeRagasResult:
        self.last_dataset = dataset
        self.last_kwargs = kwargs
        if self.raise_on_evaluate is not None:
            raise self.raise_on_evaluate
        n = len(dataset)
        scores = {
            "faithfulness": [0.9] * n,
            "answer_relevancy": [0.8] * n,
            "context_precision": [0.7] * n,
            "context_recall": [0.6] * n,
        }
        return _FakeRagasResult(scores)


def test_build_dataset_translates_examples():
    """``_build_dataset`` maps RAGHub examples to the ragas schema."""
    from raghub.eval.ragas import RagasAdapter

    adapter = RagasAdapter.__new__(RagasAdapter)  # bypass __init__ (no ragas)
    examples = [
        {
            "question": "q1",
            "answer": "a1",
            "contexts": ["c1", "c2"],
            "ground_truth": "g1",
        },
        {
            "question": "q2",
            "answer": "a2",
            "contexts": [],
            "ground_truth": "",
        },
    ]
    dataset = adapter._build_dataset(examples)
    rows = list(dataset)
    assert len(rows) == 2
    assert rows[0]["question"] == "q1"
    assert rows[0]["contexts"] == ["c1", "c2"]
    assert rows[0]["ground_truth"] == "g1"
    assert rows[1]["contexts"] == []


def test_build_dataset_handles_missing_fields():
    """``_build_dataset`` supplies defaults for missing fields."""
    from raghub.eval.ragas import RagasAdapter

    adapter = RagasAdapter.__new__(RagasAdapter)
    rows = list(adapter._build_dataset([{"question": "q"}]))
    assert rows[0]["answer"] == ""
    assert rows[0]["contexts"] == []
    assert rows[0]["ground_truth"] == ""


def test_extract_scores_returns_per_row_list():
    """``_extract_scores`` returns a list per metric, in row order."""
    from raghub.eval.ragas import RagasAdapter

    fake = _FakeRagasResult(
        {
            "faithfulness": [1.0, 0.5, 0.0],
            "answer_relevancy": [0.8, 0.6, 0.4],
            "context_precision": [0.7, 0.7, 0.7],
            "context_recall": [0.3, 0.3, 0.3],
        }
    )
    scores = RagasAdapter._extract_scores(fake, 3)
    assert scores["faithfulness"] == [1.0, 0.5, 0.0]
    assert scores["answer_relevancy"] == [0.8, 0.6, 0.4]
    assert len(scores["context_precision"]) == 3


def test_extract_scores_zeros_missing_metrics():
    """A metric missing from the result is treated as all zeros."""
    from raghub.eval.ragas import RagasAdapter

    fake = _FakeRagasResult({})  # no metrics at all
    scores = RagasAdapter._extract_scores(fake, 5)
    assert scores["faithfulness"] == [0.0] * 5
    assert scores["answer_relevancy"] == [0.0] * 5


def test_extract_scores_handles_corrupt_values():
    """A metric whose values can't be cast to float falls back to zeros."""
    from raghub.eval.ragas import RagasAdapter

    fake = _FakeRagasResult({"faithfulness": ["nope", "still nope"]})
    scores = RagasAdapter._extract_scores(fake, 2)
    assert scores["faithfulness"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Constructor + import-time behavior
# ---------------------------------------------------------------------------


def test_ragas_adapter_raises_missing_dep_when_ragas_not_installed(monkeypatch):
    """The constructor raises MissingDep when ragas is not importable."""
    # Simulate ragas not being installed by injecting an import
    # failure into sys.modules.
    import sys

    monkeypatch.delitem(sys.modules, "ragas", raising=False)
    monkeypatch.setitem(sys.modules, "ragas", None)

    from raghub.eval.ragas import RagasAdapter

    with pytest.raises(MissingDep, match="ragas"):
        RagasAdapter()


def test_ragas_adapter_raises_config_error_for_unknown_metric():
    """The constructor raises ConfigurationError on unknown metric names."""
    import sys

    # Install a stub ragas so the import succeeds, then verify the
    # metric registry rejects a bad name.
    class _StubFaithfulness:
        pass

    fake_ragas = type(sys)("ragas")
    fake_ragas_metrics = type(sys)("ragas.metrics")
    fake_ragas_metrics.faithfulness = _StubFaithfulness()
    fake_ragas_metrics.answer_relevancy = _StubFaithfulness()
    fake_ragas_metrics.context_precision = _StubFaithfulness()
    fake_ragas_metrics.context_recall = _StubFaithfulness()
    fake_ragas.metrics = fake_ragas_metrics

    sys.modules["ragas"] = fake_ragas
    sys.modules["ragas.metrics"] = fake_ragas_metrics

    try:
        from raghub.eval.ragas import RagasAdapter

        with pytest.raises(ConfigurationError, match="Unknown ragas metric"):
            RagasAdapter(metrics=["nonexistent"])
    finally:
        for name in ("ragas", "ragas.metrics"):
            sys.modules.pop(name, None)
        # Reload real module
        from importlib import reload

        import raghub.eval.ragas as _pkg
        reload(_pkg)


# ---------------------------------------------------------------------------
# evaluate() path — only runnable when ragas is installed
# ---------------------------------------------------------------------------


def _safe_import_ragas() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


# Skip only the tests that actually invoke ragas. The translation
# helpers above cover the rest.
RAGAS_AVAILABLE = _safe_import_ragas()


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_calls_ragas_evaluate_with_metric_instances(monkeypatch):
    """``evaluate()`` builds the dataset and invokes ragas.evaluate."""
    from raghub.eval.ragas import RagasAdapter

    fake_ragas = _FakeRagasModule()
    monkeypatch.setattr("raghub.eval.ragas._import_ragas", lambda: fake_ragas)

    # Bypass the __init__ import check; set the bits the adapter
    # actually uses.
    adapter = RagasAdapter.__new__(RagasAdapter)
    adapter.metric_names = ("faithfulness", "answer_relevancy")
    adapter.llm = None
    adapter.embeddings = None
    adapter._metric_instances = [object(), object()]

    async def factory(example):
        return ("the answer", ["ctx"], ["id1"], ["id1"])

    rows = [
        {"question": "q1", "answer": "a1", "contexts": ["c1"], "ground_truth": "g1"},
        {"question": "q2", "answer": "a2", "contexts": ["c2"], "ground_truth": "g2"},
    ]

    results = adapter.evaluate(rows, response_factory=factory)
    import asyncio
    results = asyncio.run(results)

    assert len(results) == 2
    assert fake_ragas.last_kwargs is not None
    assert "metrics" in fake_ragas.last_kwargs
    assert len(fake_ragas.last_kwargs["metrics"]) == 2
    # The factory's answers should have replaced the dict answers.
    assert fake_ragas.last_dataset[0]["answer"] == "the answer"


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_wraps_ragas_failure_in_configuration_error(monkeypatch):
    """A ragas exception surfaces as ConfigurationError."""
    from raghub.eval.ragas import RagasAdapter

    fake_ragas = _FakeRagasModule(raise_on_evaluate=RuntimeError("boom"))
    monkeypatch.setattr("raghub.eval.ragas._import_ragas", lambda: fake_ragas)

    adapter = RagasAdapter.__new__(RagasAdapter)
    adapter.metric_names = ("faithfulness",)
    adapter.llm = None
    adapter.embeddings = None
    adapter._metric_instances = [object()]

    async def factory(example):
        return ("a", ["c"], ["id1"], ["id1"])

    rows = [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}]
    import asyncio

    with pytest.raises(ConfigurationError, match="ragas evaluation failed"):
        asyncio.run(adapter.evaluate(rows, response_factory=factory))


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_mark_pass_when_all_metrics_above_threshold():
    """The ``passed`` flag is True when every metric is >= 0.5."""
    from raghub.eval.ragas import RagasAdapter

    fake_ragas = _FakeRagasModule()
    fake_ragas.evaluate = lambda dataset, **_: _FakeRagasResult(
        {
            "faithfulness": [0.9, 0.6],
            "answer_relevancy": [0.8, 0.7],
            "context_precision": [0.7, 0.7],
            "context_recall": [0.6, 0.6],
        }
    )

    adapter = RagasAdapter.__new__(RagasAdapter)
    adapter.metric_names = ("faithfulness", "answer_relevancy")
    adapter.llm = None
    adapter.embeddings = None
    adapter._metric_instances = [object(), object()]

    async def factory(ex):
        return ("a", ["c"], ["id"], ["id"])

    rows = [
        {"question": "q1", "answer": "a1", "contexts": ["c"], "ground_truth": "g"},
        {"question": "q2", "answer": "a2", "contexts": ["c"], "ground_truth": "g"},
    ]
    import asyncio
    results = asyncio.run(adapter.evaluate(rows, response_factory=factory))
    assert all(r.passed for r in results)


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_mark_fail_when_any_metric_below_threshold():
    """The ``passed`` flag is False when any metric is < 0.5."""
    from raghub.eval.ragas import RagasAdapter

    fake_ragas = _FakeRagasModule()
    fake_ragas.evaluate = lambda dataset, **_: _FakeRagasResult(
        {
            "faithfulness": [0.3],
            "answer_relevancy": [0.9],
            "context_precision": [0.9],
            "context_recall": [0.9],
        }
    )

    adapter = RagasAdapter.__new__(RagasAdapter)
    adapter.metric_names = ("faithfulness", "answer_relevancy")
    adapter.llm = None
    adapter.embeddings = None
    adapter._metric_instances = [object(), object()]

    async def factory(ex):
        return ("a", ["c"], ["id"], ["id"])

    rows = [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}]
    import asyncio
    results = asyncio.run(adapter.evaluate(rows, response_factory=factory))
    assert not results[0].passed


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_metric_names_have_ragas_prefix():
    """The metric keys in EvaluationResult are prefixed with ``ragas_``."""
    from raghub.eval.ragas import RagasAdapter

    adapter = RagasAdapter.__new__(RagasAdapter)
    adapter.metric_names = ("faithfulness", "answer_relevancy")
    adapter.llm = None
    adapter.embeddings = None
    adapter._metric_instances = [object(), object()]

    async def factory(ex):
        return ("a", ["c"], ["id"], ["id"])

    rows = [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}]
    import asyncio
    results = asyncio.run(adapter.evaluate(rows, response_factory=factory))
    assert "ragas_faithfulness" in results[0].metrics
    assert "ragas_answer_relevancy" in results[0].metrics
    assert "faithfulness" not in results[0].metrics  # bare name not present


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_benchmark_attribute():
    """The adapter's ``benchmark`` attribute is ``ragas``."""
    from raghub.eval.ragas import RagasAdapter

    assert RagasAdapter.benchmark == "ragas"


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
def test_evaluate_default_metrics_constant():
    """``DEFAULT_METRICS`` lists the four canonical ragas metrics."""
    from raghub.eval.ragas import RagasAdapter

    assert RagasAdapter.DEFAULT_METRICS == (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
