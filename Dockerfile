# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create a non-root user and group.
RUN groupadd --system raghub && useradd --system --gid raghub --home /app raghub

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -e ".[api,ui,dev]"

# Copy the application code.
COPY raghub ./raghub
COPY config ./config
COPY streamlit_app.py ./

# Drop privileges.
RUN chown -R raghub:raghub /app
USER raghub

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -m raghub.cli health || exit 1

CMD ["uvicorn", "raghub.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
