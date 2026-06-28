# Development Guide

## Environment Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Running Tests

```bash
python -m pytest tests/ -v            # all tests
python -m pytest tests/ -x            # stop on first failure
python -m pytest tests/ -xvs          # verbose, stop on first failure
python -m pytest tests/test_platform.py::test_ingest_and_query_isolated_access -xvs
```

FinanceBench evaluation (requires `FINANCEBENCH_EVAL=1`):

```bash
FINANCEBENCH_EVAL=1 python -m pytest tests/test_financebench.py -xvs
```

Or run the standalone eval script:

```bash
python evaluate_financebench.py
```

## Code Style

- Google Python Style Guide
- No underscore prefix on public names (no `_conn`, `_run`, etc.)
- No inline comments in code
- Type hints everywhere
- Active Record domain pattern for models

## Linting & Type Checking

```bash
ruff check raghub/ tests/
mypy raghub/
```

The codebase is clean: zero ruff errors and zero mypy errors in `raghub/`.

## Adding a New Parser

1. Create `raghub/documents/parsers/<format>_parser.py` extending `FileParser`
2. Register in `ParserRegistry.__init__`
3. Add optional dependency to `requirements/parsers.txt`

## Adding a New Embedding Provider

1. Create `raghub/embeddings/<name>.py` extending `BaseEmbeddingProvider`
2. Implement `embed_text()` and `embed_texts()`
3. Add factory logic to `raghub/embeddings/__init__.py`

## Adding a New LLM Provider

1. Create `raghub/llm/<name>.py` extending `BaseLLMProvider`
2. Implement `generate()`
3. Add factory logic to `raghub/llm/__init__.py`

## Resetting the Environment

```bash
./cleanup.sh      # removes .venv, data/, __pycache__
./setup.sh        # rebuild from scratch
```
