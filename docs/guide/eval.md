# Evaluation

`@revex/eval` measures retrieval and answer quality. It provides metrics, an
LLM-as-judge, CARE scoring, lost-in-the-middle probing, a run harness, synthetic
data generation, and a release gate.

## Metrics

`@revex/core`-agnostic functions take a `RankedHit[]`, ground-truth IDs, and
optional answer/reference/context:

| Function | Signature (paraphrased) | Description |
|---|---|---|
| `recallAtK` | `(opts) => number` | Fraction of ground-truth IDs in top-K hits. |
| `precisionAtK` | `(opts) => number` | Fraction of top-K hits that are ground truth. |
| `mrr` | `(opts) => number` | Mean reciprocal rank of first ground-truth hit. |
| `contextRecall` | alias | Same as `recallAtK`. |
| `contextPrecision` | alias | Same as `precisionAtK`. |
| `faithfulness` | `(opts) => number` | Fraction of `contextClaims` found in `contextText`. |
| `answerCorrectness` | `(opts) => number` | 1 exact, 0.5 substring, 0 mismatch. |
| `computeMetrics` | `(opts) => Metrics` | All 7 metrics in one object. |

`MetricOptions`: `hits`, `groundTruthIds`, `answer`, `reference`,
`contextClaims`, `contextText`, `k`.

## Judge

`judge({ llm, model }, { question, answer, reference, context })` uses an
LLM-as-judge to return `{ correctness, faithfulness, reasoning }`.

## CARE

`judgeCare({ llm, model, list, question, goldAnswer, signal })` labels each
retrieved chunk RELEVANT/NOT_RELEVANT, trying the LLM first and falling back to
a deterministic heuristic. `careMetrics(labels)` computes `precision`,
`recall`, and `f1`.

## Lost in the middle

`lostInMiddleProbe({ llm, model, goldChunkIds, candidateChunks, listSize,
query, reference })` places the gold chunk at each position and returns one
`LimSample { position, accuracy }` per position, measuring positional bias.

## Harness

- `runSamples(samples, { retrieval, llm, model, k })` — runs
  retrieval + generation per sample and computes metrics.
- `aggregate(results)` — averages into `AggregateMetrics`.
- `loadJsonl(raw)` / `loadJsonlFile(path)` — parse `QASample[]` (skips
  malformed lines).
- `runFinance(samples, opts)` — Finance benchmark (`financebench`) run.
- `runFrames(samples, opts)` — FRAMES benchmark run.

`QASample`: `{ id, question, goldAnswer, goldIds, metadata? }`.
`SampleResult`: `{ sample, metrics, latencyMs, answer }`.

## Synthetic data

`generateSynthetic(passages, { llm, model, count })` produces `count` QA pairs
from source passages.

## Gate

`evaluateGate(metrics, thresholds)` compares aggregate metrics against
`GateThresholds` (`recallAtK?`, `precisionAtK?`, `mrr?`, `faithfulness?`,
`answerCorrectness?`) and returns `{ status: 'pass' | 'fail', failures, metrics }`.

## CLI

`revex eval finance|frames -i <jsonl>` runs the harness against a JSONL file
(using a stub retrieval/LLM in the CLI); `revex eval lost-in-middle` is a
placeholder probe.