# Docstring Style — Google

Every public function, method, class, and module ships a
Google-style docstring. Style is enforced by `interrogate --fail-under=100`.

## Module docstring

The first line is a single-sentence summary. Subsequent paragraphs
provide context. The summary fits on one line:

```python
"""Retrieval support: rerankers, query transformers, fusion, faceted search.

The module package surface:

    Identity          - no-op pass-through for the rerank stage.
    RerankerFactory    - build rerankers from application Settings.

Long-context LLM rerank is exposed as :class:`Context` (was the
``LongContextRerankPass`` module)."""
```

## Function / method docstring

Single-line summary line, then blank, then sections. Sections used:

- `Args:`   – argument-by-argument description
- `Returns:` – return-value description
- `Raises:` – exception types and conditions
- `Note:`   – caveats, edge cases
- `Example:` – optional usage example
- `Yields:` – for generators

```python
def verify(self, source_chunks: list[Chunk]) -> None:
    """Assert the chunk's invariant contract.

    Confirms sha256(text) matches ``checksum`` and walks every
    nested child entity.

    Args:
        source_chunks: The candidate source chunks to cross-check
            against this chunk's citation graph.

    Raises:
        VerificationError: When the checksum doesn't match or a
            child entity fails its own ``verify()``.
    """
```

## Class docstring

One-line summary; *Args* is implicit (the `__init__` docstring
covers it). Document public methods inline:

```python
class Chunk:
    """Storage chunk with text and a SHA-256 content checksum.

    The chunk is the smallest unit of retrieval. Each chunk carries
    a stable ``id`` and a checksum that the producer must keep in
    sync with ``text``; :meth:`verify` enforces that invariant.

    Attributes:
        id: Stable chunk id (UUID).
        text: The actual text content of the chunk.
        checksum: SHA-256 of ``text`` (hex-digest).
    """
```

## Imperative mood

The summary line uses **imperative mood** ("Compute X" rather than
"Computes X" or "An instance that..."). Verb-first is the rule; ``ruff``
D401 enforces it.

## Periods

The summary line ends with a period. `ruff` D415 enforces it.

## One-line summaries

For trivial public classes and functions, a single-line docstring
is fine:

```python
def chunk_id_for(record_id: str) -> str:
    """Return the canonical chunk id for ``record_id``."""
```

Multi-line summaries that have body text MUST start with a blank
line after the summary. `ruff` D205 enforces this.

## When not to use Google style

- One-time lambda / factory functions that are obvious from name:
  `factory_b = lambda: ...` — these are local in tests.
- Test fixtures — pytest discovers them by name; docstrings are noise
  unless they explain behaviour.

These still need a docstring if they take parameters or raise; the
Google style applies.

## Examples

`interrogate -c pyproject.toml` runs at every CI step. `make docstrings`
runs the same command locally.
