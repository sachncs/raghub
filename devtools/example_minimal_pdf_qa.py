"""Minimal example: ingest a file and ask a question with fewer than 20 lines.

Run with::

    python examples/minimal_pdf_qa.py ./path/to/document.pdf "What is the revenue guidance?"

The script auto-falls-back to the in-memory vector store and the
heuristic LLM when external services (Qdrant, LiteLLM, OpenAI) are
unavailable, so it always runs end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

from raghub import RAG


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md")
    question = sys.argv[2] if len(sys.argv) > 2 else "Summarise the document."
    rag = RAG()
    rag.ingest(path)
    response = rag.query(question)
    print(response.answer)


if __name__ == "__main__":
    main()
