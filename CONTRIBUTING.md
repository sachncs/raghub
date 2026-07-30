# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

- Google Python Style Guide
- No underscore prefix on public names
- No inline comments in code
- Type hints everywhere
- Single-word class names preferred; protocols use a `Protocol` suffix
  when they collide with concrete classes (`GeneratorProtocol`,
  `ToolProtocol`, `SessionStoreProtocol`)

## Before Submitting

1. Run tests: `pytest tests/ -q`
2. Run linter: `ruff check raghub/ tests/`
3. Run coverage gate: `pytest tests/ --cov=raghub --cov-fail-under=85`
4. CI runs on push to `master` and on PRs; both jobs must pass.

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes in atomic commits
3. Ensure all checks pass locally
4. Open a PR against `master`
