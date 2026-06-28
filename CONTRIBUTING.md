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

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all checks pass
4. Open a PR against `main`
