.PHONY: help test lint typecheck format coverage bench docs security audit audit-local clean install dev-api dev-ui db-init db-reset

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
	@echo "  make audit-local - Run pip-audit on installed packages (no requirements/*.txt)"
	@echo "  make docs        - Build the documentation"
	@echo "  make bench       - Run the performance benchmark"
	@echo "  make dev-api     - Start the API development server (uvicorn)"
	@echo "  make dev-ui      - Start the Streamlit UI"
	@echo "  make db-init     - Initialise database tables"
	@echo "  make db-reset    - Reset database (drop + recreate)"
	@echo "  make clean       - Remove build artefacts"

install:
	pip install -e ".[api,ui,dev]"

test:
	pytest -q

coverage:
	pytest -q --ignore=tests/test_financebench.py \
		--cov=raghub --cov-report=term-missing --cov-fail-under=90

lint:
	ruff check raghub/ tests/ examples/ bench/

typecheck:
	mypy raghub/

format:
	ruff format raghub/ tests/ examples/ bench/

security:
	bandit -r raghub/ -q -ll -i

audit:
	pip-audit || true

audit-local:
	pip-audit --ignore-dev || true

docs:
	mkdocs build --strict

bench:
	python -m bench.benchmark --documents 5 --queries 20 --concurrency 4

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

dev-api:
	uvicorn raghub.api.app:app --host 0.0.0.0 --port 8000 --reload

dev-ui:
	streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

db-init:
	python -c \
		"from raghub.core.container import build_application; import asyncio; asyncio.run(build_application())"

db-reset:
	rm -f data/*.db data/sessions.json data/registry.json
	python -c \
		"from raghub.core.container import build_application; import asyncio; asyncio.run(build_application())"
