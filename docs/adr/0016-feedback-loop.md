# ADR 0016 — Feedback loop

## Status

Accepted (v0.7.7)

## Context

Users need a way to indicate whether a retrieval result was useful so
that future queries surface better chunks. Without a feedback signal the
system cannot adapt to domain-specific relevance judgements.

## Decision

We ship two feedback-driven scoring algorithms that modify BM25 and
vector scores at query time:

### `bm25-boost`

When a chunk receives positive feedback (`score > 0`), its BM25 term
frequencies are scaled up by a multiplicative boost:

```
boosted_tf = tf × (1 + α × feedback_score)
```

where `α` (default `0.5`) is the `RAGHUB_BM25_BOOST_ALPHA` tunable.
The feedback score is the rolling average of all user-provided scores
for that chunk, clamped to `[0, 1]`.

### `vector-down-weight`

When a chunk receives negative feedback (`score < 0`), its cosine
similarity score is damped:

```
adjusted_score = similarity × (1 + β × feedback_score)
```

where `β` (default `0.3`) is the `RAGHUB_VECTOR_DOWN_WEIGHT_BETA`
tunable and `feedback_score` is in `[-1, 0]`.

Both adjustments are applied **after** the base retrieval step and
**before** reranking.

## Consequences

- **No training loop**: These are score-level heuristics; the
  underlying embeddings and BM25 index are not retrained.
- **Transparency**: The `FeedbackStore` records every score with a
  timestamp, making audits straightforward.
- **Tuning required**: `α` and `β` must be tuned per domain; defaults
  are conservative.

## Alternatives considered

- **Always-on retraining**: Deferred — would require a background
  worker and embedding recomputation, too heavy for v0.7.x.
- **RLHF**: Out of scope for this release; could be layered on later
  via the same `FeedbackStore`.
- **Hard filter (remove negatively-scored chunks)**: Rejected — too
  aggressive; a chunk may be irrelevant for one query but useful for
  another.
