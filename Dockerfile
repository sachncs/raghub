FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY raghub ./raghub
COPY config ./config
COPY streamlit_app.py ./

RUN pip install --no-cache-dir -e ".[api,ui]"

EXPOSE 8000 8501

CMD ["python", "-m", "raghub.cli", "health"]

