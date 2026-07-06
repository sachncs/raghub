# Contributing

## Development Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Code Style

- Google Python Style Guide
- No underscore prefix on public names
- No inline comments in code
- Type hints everywhere

## Before Submitting

1. Run tests: `python -m pytest tests/ -v`
2. Run linter: `ruff check raghub/ tests/`
3. Run type checker: `mypy raghub/`
4. Ensure all FinanceBench tests pass: `FINANCEBENCH_EVAL=1 python -m pytest tests/test_financebench.py -xvs`

## Test Environment Variables

The following environment variables gate optional / slow test suites.
Set them only when you want the corresponding tests to run; they
default to **off** so the local `pytest` run stays fast.

| Variable                     | Effect                                                                 |
|------------------------------|------------------------------------------------------------------------|
| `RAGHUB_RUN_PLATFORM_TESTS=1` | Runs `tests/test_platform.py` (live LLM + vector store round-trips).    |
| `FINANCEBENCH_EVAL=1`         | Runs the `tests/test_financebench.py` evaluation suite against the downloaded FinanceBench dataset. |

Example:

```bash
RAGHUB_RUN_PLATFORM_TESTS=1 pytest -q tests/test_platform.py
FINANCEBENCH_EVAL=1 pytest -q tests/test_financebench.py
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all checks pass
4. Open a PR against `main`
