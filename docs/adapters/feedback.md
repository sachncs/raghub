# Feedback Store

The feedback store persists user feedback (thumbs up/down + comments)
for retrieval-quality evaluation.

## Registration

```python
from raghub.feedback.core import FeedbackStore, SqliteFeedbackStore
```

`FeedbackStore` is the Registry base. `SqliteFeedbackStore` is registered
as `@FeedbackStore.register("sqlite")`.

## Usage

```python
from raghub.feedback.core import FeedbackStore, SqliteFeedbackStore

store = SqliteFeedbackStore(db_path="feedback.db")
await store.initialize()
await store.record(feedback)
results = await store.aggregate(tenant_id="acme")
```

## Features

- `record()` persists feedback with redacted comments.
- `aggregate()` returns per-tenant rating counts.
- `list_for_session()`, `list_for_chunk()`, `list_for_tenant()` for filtering.
- Comment redaction strips PII before storage.

## Adding a Backend

Implement `FeedbackStore` and register with `@FeedbackStore.register("name")`:

```python
from raghub.feedback.core import FeedbackStore

@FeedbackStore.register("pg")
class PgFeedbackStore(FeedbackStore):
    ...
```
