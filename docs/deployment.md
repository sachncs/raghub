# Deployment

## Local Development

```bash
pip install -e ".[api,ui,dev]"
uvicorn raghub.api.app:app --reload
streamlit run streamlit_app.py
```

## Docker

Use the provided `Dockerfile` and `docker-compose.yml`.

## Configuration

Profiles:

- `config/development.yaml`
- `config/staging.yaml`
- `config/production.yaml`

Environment variables override profile values.

