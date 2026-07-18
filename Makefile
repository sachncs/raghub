.PHONY: help install test coverage lint typecheck format security audit docs bench dev-api dev-ui db-init db-reset clean compose-build compose-up compose-down compose-dev compose-dev-down compose-prod compose-prod-down

help:
	@echo "Common targets:"
	@echo "  make install          - Install the package in editable mode"
	@echo "  make test             - Run the test suite"
	@echo "  make coverage         - Run the test suite with coverage"
	@echo "  make lint             - Run ruff"
	@echo "  make typecheck        - Run mypy"
	@echo "  make format           - Auto-format with ruff"
	@echo "  make security         - Run bandit"
	@echo "  make audit            - Run pip-audit (fails on vulnerabilities)"
	@echo "  make docs             - Build the documentation"
	@echo "  make bench            - Run the performance benchmark"
	@echo "  make dev-api          - Start the API dev server (uvicorn --reload)"
	@echo "  make dev-ui           - Start the Streamlit UI"
	@echo "  make db-init          - Initialise database tables"
	@echo "  make db-reset         - Reset database (drop + recreate)"
	@echo "  make compose-build    - Build production images"
	@echo "  make compose-up       - Start production stack (api + ui + qdrant)"
	@echo "  make compose-down     - Stop production stack"
	@echo "  make compose-dev      - Start dev stack with --reload and source mounts"
	@echo "  make compose-dev-down - Stop dev stack"
	@echo "  make clean            - Remove build artefacts"

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
	pip-audit

docs:
	mkdocs build --strict

bench:
	python -m bench.benchmark --documents 5 --queries 20 --concurrency 4

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

dev-api:
	uvicorn raghub.api.app:get_app --factory --host 0.0.0.0 --port 8000 --reload

dev-ui:
	streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

db-init:
	python -c \
		"from raghub.core.container import build_application; import asyncio; asyncio.run(build_application())"

db-reset:
	rm -f data/*.db data/sessions.json data/registry.json
	python -c \
		"from raghub.core.container import build_application; import asyncio; asyncio.run(build_application())"

# ---------- docker compose ----------

compose-build:
	docker compose -f docker-compose.yml build

compose-up:
	docker compose -f docker-compose.yml --profile production up -d

compose-down:
	docker compose -f docker-compose.yml --profile production down

compose-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev up

compose-dev-down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev down
