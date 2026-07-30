"""evaluation package.

Implementation lives in :mod:`raghub.helper` (evaluation); local entry-point modules: ['cli'].
"""

from __future__ import annotations

# --- cli.py content ---
import asyncio
import statistics
from typing import Any

import typer

from raghub.helper.eval import (
    TOKEN_RE,
    FinanceBench,
    FramesBenchmark,
    Metrics,
    Scoring,
    run,
)
from raghub.utils import write_json

app = typer.Typer(help="Evaluation harnesses.", no_args_is_help=True)


@app.command(name="financebench")
def financebench(
    examples: int = typer.Option(10, "--examples", "-n", help="Number of examples (0 = all)."),
) -> None:
    """Run the FinanceBench evaluator and print a JSON summary."""
    async def runner() -> None:
        evaluator = FinanceBench()
        examples_list: list[dict[str, Any]] = []
        if examples:
            rows = await asyncio.to_thread(evaluator.ensure_examples)
            examples_list.extend(rows[:examples])

        async def factory(_example: object) -> str:
            return ""

        results = await run(evaluator, examples_list, response_factory=factory)
        summary = {
            "benchmark": evaluator.benchmark,
            "count": len(results),
            "pass_rate": statistics.mean(1.0 if r.passed else 0.0 for r in results)
            if results
            else 0.0,
            "metrics": {
                name: statistics.mean(r.metrics.get(name, 0.0) for r in results)
                for name in {k for r in results for k in r.metrics}
            },
        }
        write_json(
            {
                "summary": summary,
                "results": [r.model_dump(mode="json") for r in results],
            }
        )

    asyncio.run(runner())


@app.command(name="frames")
def frames(
    examples: int = typer.Option(0, "--examples", "-n", help="Number of FRAMES examples (0 = all 824)."),
) -> None:
    """Run the FRAMES evaluator and print a JSON summary."""
    async def runner() -> None:
        evaluator = FramesBenchmark()
        examples_list: list[dict[str, Any]] = []
        rows = await asyncio.to_thread(evaluator.ensure_examples)
        examples_list.extend(rows if examples == 0 else rows[:examples])

        async def factory(_example: object) -> str:
            return ""

        results = await run(evaluator, examples_list, response_factory=factory)
        summary = {
            "benchmark": evaluator.benchmark,
            "count": len(results),
            "pass_rate": statistics.mean(1.0 if r.passed else 0.0 for r in results)
            if results
            else 0.0,
            "metrics": {
                name: statistics.mean(r.metrics.get(name, 0.0) for r in results)
                for name in {k for r in results for k in r.metrics}
            },
        }
        write_json(
            {
                "summary": summary,
                "results": [r.model_dump(mode="json") for r in results],
            }
        )

    asyncio.run(runner())



__all__ = ['TOKEN_RE', 'FinanceBench', 'FramesBenchmark', 'Metrics', 'Scoring', 'run']
