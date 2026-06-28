# Future Extensions

The framework is designed to accept new adapters and policies without changing business logic.

## Embedding Providers

- OpenAI (text-embedding-3-small/large)
- Jina, Nomic, Cohere
- BGE (BAAI General Embedding)

## LLM Providers

- OpenAI (GPT-4, GPT-4o)
- Anthropic Claude
- Google Gemini
- Ollama (local)

## Vector Stores

- Milvus, Qdrant, Pinecone
- FAISS, Chroma
- pgvector (PostgreSQL)

## Infrastructure

- Background task queues (Celery, RQ, or Arq)
- ABAC policies over document and user attributes
- Multi-tenant database isolation
- Caching layer (Redis) for embeddings and LLM responses
