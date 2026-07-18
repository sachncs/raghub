"""Interface-first contracts for replaceable dependencies.

The :mod:`raghub.interfaces` package defines :class:`typing.Protocol`
classes that describe the public surface of each replaceable
dependency (LLM, embeddings, retrieval, storage, observability,
prompt builder, workers). Concrete implementations live in their own
sub-packages; tests and alternate deployments can substitute their
own types without depending on the production stack.

Each sub-module groups one logical family of protocols:

* :mod:`.embeddings` — embedding provider contract.
* :mod:`.llm` — LLM provider contract.
* :mod:`.observability` — logger + metrics contracts.
* :mod:`.prompts` — prompt builder contract.
* :mod:`.retrieval` — retriever + reranker contracts.
* :mod:`.storage` — document registry, conversation store, session
  store contracts.
* :mod:`.vectorstore` — vector store contract.
* :mod:`.workers` — background worker + task queue contracts.
"""
