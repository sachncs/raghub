"""Custom-component example showing how to swap every dependency.

The :class:`RAG` constructor accepts concrete components for any
spec-defined interface, allowing the framework to be embedded in
existing services with bespoke embedders, vector stores, or
generators.
"""

from __future__ import annotations

from raghub import RAG
from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.llm.heuristic import HeuristicLLMProvider
from raghub.vectorstore.memory import InMemoryVectorStore


class LoggingEmbedder(HashingEmbeddingProvider):
    def embed_text(self, text: str) -> list[float]:
        print(f"embed: {text[:32]!r}")
        return super().embed_text(text)


rag = RAG(
    embedder=LoggingEmbedder(dimension=128, model_name="hashing-custom"),
    vector_store=InMemoryVectorStore(),
    llm=HeuristicLLMProvider(),
)

rag.ingest("README.md")
print(rag.query("What is this project?").answer)
