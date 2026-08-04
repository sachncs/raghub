# Evaluation Guide

This guide covers how to evaluate a RAGHub deployment end-to-end —
from building a golden dataset, through running benchmarks, to
setting quality gates that fail CI when metrics drop.

## The eval pipeline at a glance

```
[AnyScale doc]
1. Build a golden dataset
2. Run the pipeline against it
3. Measure retrieval quality + generation quality
4. Set thresholds + alerts
5. Run automatically in CI
```

A "golden dataset" is a small set of (question, expected_response,
relevant_documents) triples that you've manually verified. The
dataset is the ground truth against which every metric is
computed.

## 1. Building a golden dataset

Two approaches: hand-craft from domain experts, or generate from
your corpus.

### Option A: Hand-craft from domain experts

The most reliable. Create a JSONL file with one record per line:

```json
{"question": "What was the Q3 2024 revenue?", "answer": "12.4 million", "contexts": ["Q3 2024 revenue was 12.4 million USD"], "relevant_ids": ["doc-42"]}
```

Save it as `golden.jsonl` and check it in. Domain experts should
review a sample weekly to catch dataset drift.

### Option B: Generate from your corpus

Use `SyntheticDataset` to produce (question, answer, contexts)
triples from your ingested documents. This is the AnyScale doc's
"synthetic test-set generation" recommendation.

```python
from raghub.eval import SyntheticDataset
from raghub.llm import LiteLLM

ds = SyntheticDataset(
    corpus=rag.vector_store.records,  # optional: filter to your KB
    llm=LiteLLM(model="gpt-4o-mini"),
    n_questions=100,
    seed=42,  # reproducible runs
)
examples = await ds.generate()

# Save to disk for re-use
import json
with open("golden.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")
```

The generator needs a real LLM. With the offline `HeuristicProvider`
(no API key), it produces a degenerate but valid dataset — useful
for testing the pipeline, not for measuring real quality.

Limitations: synthetic data inherits the generator's biases and
lacks the long-tail complexity of real queries. Always pair
synthetic generation with a small hand-curated validation set.

## 2. Running a benchmark

RAGHub ships with two datasets built in:

- **Finance** — finance-domain Q&A from PatronusAI.
- **Frames** — multi-hop reasoning from Google.

Both load from HuggingFace Hub with local caching. The
`evaluate` method takes a `response_factory` callable that maps
each example to the model's response.

```python
from raghub.eval import Finance, run

evaluator = Finance()
examples = evaluator.ensure_examples()

async def factory(example):
    return await rag.aquery(example["question"])

results = await run(evaluator, examples, response_factory=factory)
```

The benchmark smoke test in `tests/test_benchmark_smoke.py` shows
the offline test pattern — no HuggingFace download, just an in-
memory dataset + a stub retriever.

## 3. Metrics

`Metrics.evaluate()` returns a dict of metric name → value. The
metrics are split into retrieval and generation:

| Metric | What it measures |
| --- | --- |
| `recall_at_5` | Fraction of relevant docs in the top 5 |
| `precision_at_5` | Fraction of top 5 that is relevant |
| `f1_at_5` | Harmonic mean of precision and recall |
| `hit_rate_at_5` | 1.0 if any top-5 is relevant, else 0.0 |
| `mrr` | 1 / rank of first relevant hit |
| `map` | Mean average precision across the ranking |
| `context_recall` | Fraction of answer tokens grounded in context |
| `context_precision` | Fraction of context relevant to the question |
| `completeness` | Fraction of context tokens used in the answer |
| `coherence` | Sentence-pair topical continuity |
| `faithfulness` | Fraction of answer tokens grounded in context |
| `faithfulness_claims` | Sentence-level claim support |
| `answer_relevance` | Jaccard between question and answer content tokens |
| `answer_correctness` | Jaccard between answer and ground-truth tokens |

All metrics are bounded in `[0.0, 1.0]`. Property-based tests in
`tests/test_hypothesis_properties.py` verify this for any input.

For semantic faithfulness and relevance — the AnyScale doc's
criticism of token-overlap metrics — use `Judge`:

```python
from raghub.eval import Judge

judge = Judge(LiteLLM(model="gpt-4o-mini"))
score = await judge.faithfulness(answer, contexts)  # 0..1
score = await judge.answer_relevance(answer, question)  # 0..1
```

Judge uses two prompts that ask the model to score the answer
on a 0-1 scale. The response is parsed by `parse` (regex
extraction + clamping). Negative signs are accepted so `-0.5`
clamps to `0.0`. `max_retries` (default 1) controls the retry
budget on parse failure.

Calibrate judges against human-verified golden datasets. The
LLM-as-judge approach has known biases: preferring longer
responses, showing positional bias, favoring its own outputs.

## 4. Setting a quality gate

`Gate` raises `ConfigurationError` when any metric
breaches a threshold. Use it in CI to fail the build when
quality metrics drop.

```python
from raghub.eval import Gate
from raghub.errors import ConfigurationError

gate = Gate(
    {"recall_at_5": 0.7, "faithfulness": 0.8},
    default_mode="min",  # metric must be >= threshold
)

try:
    gate.check(metrics)
except ConfigurationError as e:
    print(f"Quality gate failed: {e}")
    sys.exit(1)
```

Add a cost metric with `mode="max"` to ensure the system stays
fast:

```python
gate = Gate(
    {"recall_at_5": 0.7, "latency_ms": 200},
    default_mode="min",
).add("latency_ms", 200, mode="max")
```

Use the fluent builder for one metric at a time:

```python
gate = (
    Gate()
    .add("recall_at_5", 0.7)
    .add("faithfulness", 0.8)
    .add("latency_ms", 200, mode="max")
)
```

For non-raising summaries (CI logs, dashboards), use `report()`:

```python
for name, (value, threshold, passed, mode) in gate.report(metrics).items():
    print(f"{name}: {value} {'>=' if mode == 'min' else '<='} {threshold} → {'PASS' if passed else 'FAIL'}")
```

## 5. A/B testing

Run two RAG instances against the same dataset and compare
metrics. Useful for measuring retrieval-quality changes
(new embedder, new chunker, new retriever).

```python
from raghub.eval import compare, Finance

result = await compare(
    rag_a=control_rag,
    rag_b=treatment_rag,
    examples=examples,
    evaluator=Finance(),
    gate=Gate({"recall_at_5": 0.7}),
)

print(f"Winner: {result['winner']}")
print(f"Metric diffs: {result['metric_diffs']}")
print(f"Gate passed: {result['gate_passed']}")
```

The winner is determined by per-metric wins: whichever RAG wins
more metrics wins the test. Ties are reported as "tie".

## 6. CI integration

The `.github/workflows/ci.yml` file has a `benchmark-smoke` job
that runs only on pushes to master:

```yaml
benchmark-smoke:
  if: github.ref == 'refs/heads/master' && github.event_name == 'push'
  steps:
    - run: pytest tests/test_benchmark_smoke.py -m benchmark -v
```

The smoke test uses an in-memory dataset and a stub retriever;
it verifies the eval pipeline runs end-to-end and that the
Gate with `recall_at_5 >= 0.5` passes. PR builds are
skipped (the job runs only on master to avoid noisy failures).

For a real benchmark run (Finance + a real retriever), run
locally:

```bash
python -m raghub eval financebench --examples 10
```

This prints a JSON summary to stdout. Add a `Gate` check
in the same script to fail CI on real metric regressions.

## 7. Best practices (from the AnyScale doc)

- **Build golden datasets.** Synthetic data is a starting point,
  not a substitute. Hand-curated examples from domain experts catch
  subtle language nuances that automated metrics miss.
- **Calibrate automated judges.** LLM-as-judge has known biases.
  Pair `Judge` scores with a small human-verified set to
  detect drift. Perfect scores on technical metrics don't
  guarantee user satisfaction.
- **Define custom metrics.** Beyond the built-in suite, you can
  add application-specific metrics (politeness, fairness across
  demographics, data leakage prevention) by extending `Metrics`.
- **Automate evaluation.** Run tests automatically when changing
  components (embedding models, prompts, retrieval parameters).
  Use `Gate` to enforce thresholds.
- **Set thresholds.** Establish minimum performance thresholds
  for key metrics. The `recall_at_5: 0.7` default is a starting
  point; tune to your domain.
- **Test security and robustness.** PII leakage, prompt injection,
  knowledge base poisoning — see `tests/test_security.py` for
  smoke tests.
- **A/B test.** Run different strategies in parallel via `compare`.
  Persist the results so you can correlate with production
  telemetry.

## 8. Limitations

The deterministic `Metrics` (token overlap, BM25) and
`Judge` (semantic, single-prompt) cover the two extremes. For
third-party evaluation frameworks (TruLens, DeepEval), integrate
via the `PluginRegistry` or call the framework directly.
