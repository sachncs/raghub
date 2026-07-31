.PHONY: help install inventory test coverage lint docstrings naming typecheck format security audit docs bench dev-api clean migrate

help:
	@echo "Common targets:"
	@echo "  make inventory       - Generate reports/inventory.json and reports/inventory.md"
	@echo "  make install         - Install the package in editable mode"
	@echo "  make test            - Run the test suite"
	@echo "  make coverage        - Run the test suite with coverage (≥85%)"
	@echo "  make lint            - Run ruff"
	@echo "  make docstrings      - Run interrogate (fail-under=100)"
	@echo "  make naming          - Run lint/naming.py (local enforcement of R2)"
	@echo "  make typecheck       - Run mypy"
	@echo "  make format          - Auto-format with ruff"
	@echo "  make security        - Run bandit"
	@echo "  make audit           - Run pip-audit"
	@echo "  make docs            - Build the documentation"
	@echo "  make bench           - Run the performance benchmark"
	@echo "  make migrate         - Migrate on-disk storage from v1 to v2"
	@echo "  make clean           - Remove build artefacts"

install:
	pip install -e ".[dev]"

inventory:
	python -m lint.inventory

test:
	pytest -q

coverage:
	pytest -q \
		--cov=raghub --cov-report=term-missing --cov-fail-under=85

lint:
	ruff check raghub/ tests/ devtools/

docstrings:
	interrogate -c pyproject.toml

naming:
	python lint/naming.py

typecheck:
	mypy raghub/

format:
	ruff format raghub/ tests/ devtools/

security:
	bandit -r raghub/ -q -ll -i

audit:
	pip-audit

docs:
	mkdocs build --strict

bench:
	python -m devtools.benchmark --documents 5 --queries 20 --concurrency 4

migrate:
	python -m raghub.migrate --root ./data

dev-api:
	raghub run --host 0.0.0.0 --port 8000

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage .coverage.*
	find . -type d -name __pycache__ -exec rm -rf {} +
