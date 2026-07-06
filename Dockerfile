# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ARG INSTALL_EXTRAS="api,ui,dev"
ARG INCLUDE_ZVEC=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create a non-root user and group.
RUN groupadd --system raghub && useradd --system --gid raghub --home /app raghub

# Leverage layer caching: copy pyproject.toml first and install deps
# so that dependency install is cached when only source code changes.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    if [ "$INCLUDE_ZVEC" = "1" ]; then \
        pip install -e ".[${INSTALL_EXTRAS},zvec]"; \
    else \
        pip install -e ".[${INSTALL_EXTRAS}]"; \
    fi

# Copy the application code.
COPY raghub ./raghub
COPY config ./config
COPY streamlit_app.py ./

# Drop privileges.
RUN chown -R raghub:raghub /app
USER raghub

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "raghub.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
