"""Eval CLI entry points.

Backs the ``raghub eval ...`` sub-app and the ``raghub-financebench``
console script. Benchmark adapters and scoring live in
:mod:`raghub.eval`.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import Any

import typer

from raghub.eval import Finance, Frames, run
from raghub.io import write_json

__all__ = ["financebench", "frames"]

app = typer.Typer(help="Evaluation harnesses.", no_args_is_help=True)


@app.command(name="financebench")
def run_financebench(
    examples: int = typer.Option(10, "--examples", "-n", help="Number of examples (0 = all)."),
) -> None:
    """Run the Finance evaluator and print a JSON summary."""

    async def runner() -> None:
        """Build examples, run the evaluator, and write the JSON summary."""
        evaluator = Finance()
        examples_list: list[dict[str, Any]] = []
        rows = await asyncio.to_thread(evaluator.ensure_examples)
        if examples == 0:
            examples_list.extend(rows)
        elif examples:
            examples_list.extend(rows[:examples])

        async def factory(_example: object) -> str:
            """Stub factory: returns the empty answer for every example."""
            await asyncio.sleep(0)
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
                "results": [r.dump(mode="json") for r in results],
            }
        )

    asyncio.run(runner())


@app.command(name="frames")
def run_frames(
    examples: int = typer.Option(
        0, "--examples", "-n", help="Number of FRAMES examples (0 = all 824)."
    ),
) -> None:
    """Run the FRAMES evaluator and print a JSON summary."""

    async def runner() -> None:
        """Build examples, run the evaluator, and write the JSON summary."""
        evaluator = Frames()
        examples_list: list[dict[str, Any]] = []
        rows = await asyncio.to_thread(evaluator.ensure_examples)
        examples_list.extend(rows if examples == 0 else rows[:examples])

        async def factory(_example: object) -> str:
            """Stub factory: returns the empty answer for every example."""
            await asyncio.sleep(0)
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
                "results": [r.dump(mode="json") for r in results],
            }
        )

    asyncio.run(runner())


financebench = run_financebench
frames = run_frames
