# Dynamic RAG Framework

Dynamic RAG is a modular Python framework for multi-user retrieval augmented generation with:

- Runtime PDF ingestion
- Metadata-aware retrieval with strict authorization-before-search
- Session-scoped conversational memory
- Pluggable embedding, vector, and LLM adapters
- FastAPI and Streamlit reference applications

## Package Layout

```text
raghub/
  api/
  auth/
  conversation/
  core/
  documents/
  embeddings/
  ingestion/
  interfaces/
  llm/
  memory/
  monitoring/
  observability/
  prompts/
  retrieval/
  services/
  storage/
  vectorstore/
```

