> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# Plugin Author Quickstart — 10 lines

```python
from raghub import Chunk, Citation, Citations, Response

class MyRetriever:
    """A retriever plugin: top-k chunks for ``question``."""

    def retrieve(self, question: str, *, top_k: int = 5) -> list[Chunk]:
        return [Chunk(id=f"c{i}", document_id="d", version=1,
                       text="x", company="", owner="",
                       checksum="b" * 64) for i in range(top_k)]


def answer(question: str, *, top_k: int = 5) -> Response:
    """Top-line answer plugin: returns a stub response with citations."""

    retriever = MyRetriever()
    chunks = retriever.retrieve(question, top_k=top_k)
    cites = Citations(items=[
        Citation(chunk=c, document_id=c.document_id, version=1,
                   page=0, section="", quote=c.text, score=0.5,
                   source_uri="mem://x")
        for c in chunks
    ])
    return Response(answer=f"Found {len(chunks)} chunks",
                    citations=cites, chunks=chunks)


assert answer("revenue").verify() is None  # entities are valid by default.
```
