"""``raghub eval financebench`` / ``raghub-financebench`` command.

Provides:

* The :func:`add_parser` hook for the top-level CLI dispatcher.
* The :func:`main` sync entry point used by the
  ``raghub-financebench`` console script.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics

from raghub.cli.common import write_json
from raghub.evaluation.financebench import FinanceBenchEvaluator


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``eval`` subcommand.

    Args:
        subparsers: The argparse subparsers container.
    """
    parser = subparsers.add_parser("eval", help="run an evaluation")
    parser.add_argument("benchmark", choices=["financebench"], help="benchmark to run")
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="number of examples to evaluate (0 = all)",
    )
    parser.set_defaults(handler=run_subcommand)


def run_subcommand(args: argparse.Namespace) -> int:
    """Run the evaluation and print a JSON summary.

    Args:
        args: The argparse namespace with ``benchmark`` and
            ``examples`` fields.

    Returns:
        ``0`` on success.
    """

    async def async_main() -> int:
        evaluator = FinanceBenchEvaluator()

        async def factory(_example: object) -> str:
            return ""

        examples = []
        if args.examples:
            rows = await asyncio.to_thread(evaluator.ensure_examples)
            examples = rows[: args.examples]
        results = await evaluator.evaluate(examples, response_factory=factory)
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
        return 0

    return asyncio.run(async_main())


def build_console_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt the console-script namespace to the subcommand's shape.

    Args:
        args: The console-script argparse namespace with only
            ``examples``.

    Returns:
        A namespace with the ``benchmark`` field populated.
    """
    return argparse.Namespace(benchmark="financebench", examples=args.examples)


def main() -> int:
    """Sync entry point for the ``raghub-financebench`` console script.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(prog="raghub-financebench")
    parser.add_argument("--examples", type=int, default=10, help="number of examples to evaluate")
    return run_subcommand(build_console_namespace(parser.parse_args()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
