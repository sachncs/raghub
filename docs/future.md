# Future Extensions

The framework is designed to accept new adapters and policies without changing business logic.

Possible next steps:

- Swap `HashingEmbeddingProvider` for BGE, OpenAI, Jina, Nomic, or Sentence Transformers.
- Swap `HeuristicLLMProvider` for OpenAI, Gemini, Claude, Azure OpenAI, or Ollama.
- Replace the in-memory/Zvec adapter with Milvus, Qdrant, Pinecone, FAISS, or Chroma adapters.
- Add background task queues such as Celery or RQ.
- Introduce ABAC policies over document and user attributes.

