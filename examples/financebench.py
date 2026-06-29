"""Run the FinanceBench evaluation against an LLM-powered RAGHub instance.

Usage::

    python examples/financebench.py
    python examples/financebench.py --examples 25

The default response factory uses the in-process :class:`RAG` facade.
The evaluator falls back to local FinanceBench JSONL if the
HuggingFace download is unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics

from raghub import RAG
from raghub.evaluation.financebench import FinanceBenchEvaluator


async def _run(examples: int) -> None:
    rag = RAG()
    evaluator = FinanceBenchEvaluator()

    async def _factory(example: dict) -> str:
        question = example.get("question") or example.get("query") or ""
        response = await rag.aquery(question)
        return response.answer

    rows = evaluator._ensure_examples()[:examples]
    results = await evaluator.evaluate(rows, response_factory=_factory)
    summary = {
        "benchmark": evaluator.benchmark,
        "count": len(results),
        "pass_rate": statistics.mean(1.0 if r.passed else 0.0 for r in results) if results else 0.0,
        "metrics": {
            name: statistics.mean(r.metrics.get(name, 0.0) for r in results)
            for name in {k for r in results for k in r.metrics}
        },
    }
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(_run(args.examples))


if __name__ == "__main__":
    main()
