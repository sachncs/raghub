# syntax=docker/dockerfile:1.7
# Build args:
#   SERVICE   - "api" (default) or "ui"; selects the container CMD and
#               the in-image healthcheck.
#   PYTHON_TAG - python:3.12-slim-bookworm by default; pin a patch tag
#               (no digest) for reproducibility.

ARG PYTHON_TAG=3.12-slim-bookworm

# ---------- builder stage ----------
FROM python:${PYTHON_TAG} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build tools needed to compile any wheels (cryptography, aiosqlite,
# bcrypt, pypdf). Slim images strip them out for the runtime layer.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY raghub ./raghub
COPY config ./config
COPY streamlit_app.py ./

# Build a wheel that the runtime stage installs with --no-deps. The
# runtime stage supplies its own dependency set via the pinned
# `requirements-runtime.txt` (generated from the project metadata so
# dev extras never ship to production).
RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /wheels \
    && python -c "import tomllib,pathlib; \
meta=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); \
extras=meta['project'].get('optional-dependencies',{}); \
runtime=sorted(set(meta['project']['dependencies'])+sum((extras.get(e,[]) for e in ('api','ui')),[])); \
pathlib.Path('/wheels/requirements-runtime.txt').write_text('\n'.join(runtime)+'\n')"

# ---------- runtime stage ----------
FROM python:${PYTHON_TAG} AS runtime

ARG SERVICE=api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_PROFILE=production \
    RAG_DATA_DIR=/app/data \
    RAG_REGISTRY_PATH=/app/data/registry.db \
    RAG_SESSIONS_PATH=/app/data/sessions.db \
    RAG_LOG_LEVEL=INFO

WORKDIR /app

# Runtime-only OS packages: tesseract (OCR via pytesseract) and
# libmagic (mime-type detection). The Python build toolchain is
# intentionally not included in this layer.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends tesseract-ocr libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root account for the running service.
RUN groupadd --system raghub \
    && useradd --system --gid raghub --home /app --shell /sbin/nologin raghub

# Install the wheel and the runtime dependency set. The wheel is
# installed first (without deps) so the runtime requirements file
# is the single source of truth for what ships.
COPY --from=builder /wheels/raghub-*.whl /wheels/raghub.whl
COPY --from=builder /wheels/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-deps /wheels/raghub.whl \
    && pip install -r /tmp/requirements-runtime.txt \
    && rm -rf /tmp/requirements-runtime.txt /wheels/raghub.whl

COPY config /app/config
RUN chown -R raghub:raghub /app

USER raghub
EXPOSE 8000 8501

# Service-aware CMD: the compose file passes the same SERVICE arg,
# so production containers run the right entry point without a
# second image variant.
RUN if [ "$SERVICE" = "ui" ]; then \
        echo '#!/bin/sh\nexec streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true' \
            > /app/entrypoint.sh; \
    else \
        echo '#!/bin/sh\nexec uvicorn raghub.api.app:get_app --factory --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips "*"' \
            > /app/entrypoint.sh; \
    fi \
    && chmod +x /app/entrypoint.sh

# Service-aware healthcheck: API hits /health; UI hits Streamlit's
# internal /_stcore/health. compose can still override per-service
# if a more specific probe is required.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD if [ "$SERVICE" = "ui" ]; then \
            curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1; \
        else \
            curl -fsS http://127.0.0.1:8000/health || exit 1; \
        fi

CMD ["/app/entrypoint.sh"]
