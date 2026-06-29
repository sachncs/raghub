.PHONY: help test lint typecheck format coverage bench docs security audit clean install

help:
	@echo "Common targets:"
	@echo "  make install     - Install the package in editable mode"
	@echo "  make test        - Run the test suite"
	@echo "  make coverage    - Run the test suite with a coverage report"
	@echo "  make lint        - Run ruff"
	@echo "  make typecheck   - Run mypy"
	@echo "  make format      - Auto-format with ruff"
	@echo "  make security    - Run bandit"
	@echo "  make audit       - Run pip-audit"
	@echo "  make docs        - Build the documentation"
	@echo "  make bench       - Run the performance benchmark"
	@echo "  make clean       - Remove build artefacts"

install:
	pip install -e ".[api,ui,dev]"

test:
	pytest -q

coverage:
	pytest -q --ignore=tests/test_financebench.py \
		--cov=raghub --cov-report=term-missing --cov-fail-under=72

lint:
	ruff check raghub/ tests/ examples/ bench/

typecheck:
	mypy raghub/

format:
	ruff format raghub/ tests/ examples/ bench/

security:
	bandit -r raghub/ -q -ll -i

audit:
	pip-audit -r requirements/all.txt || true

docs:
	mkdocs build --strict

bench:
	python -m bench.benchmark --documents 5 --queries 20 --concurrency 4

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
