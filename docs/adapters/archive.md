# Archive Store

The archive store persists ingested bundles for audit and replay.

## Registration

```python
from raghub.archive.core import ArchiveStore, LocalArchiveStore
```

`ArchiveStore` is the Registry base. `LocalArchiveStore` is registered
as `@ArchiveStore.register("local")`.

## Usage

```python
from raghub.archive.core import LocalArchiveStore

store = LocalArchiveStore(base_path="/data/archives")
store.write(bundle)
loaded = store.read(bundle_id)
```

## Features

- `write()` stores a `Bundle` with manifest and content files.
- `read()` rehydrates a bundle by ID.
- `manifest()` returns the `ArchiveManifest` for a bundle.
- `BundleComponents` groups `chunks.jsonl`, `embeddings.npy`, and `metadata.json`.

## Adding a Backend

Implement `ArchiveStore` and register with `@ArchiveStore.register("name")`:

```python
from raghub.archive.core import ArchiveStore

@ArchiveStore.register("s3")
class S3ArchiveStore(ArchiveStore):
    ...
```
