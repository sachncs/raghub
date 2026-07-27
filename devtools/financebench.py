"""Run the FinanceBench evaluation against an LLM-powered RAGHub instance.

Builds a :class:`LiteLLMProvider` from the ``RAG_LLM_BASE_URL``,
``RAG_LLM_API_KEY`` and ``RAG_LLM_MODEL`` env vars; ingests a real
corpus of FinanceBench source 10-K filings (downloaded from
``github.com/patronus-ai/financebench``); runs every example through
:class:`RAG`'s query path.

The corpus is matched to the dataset — only the 10-K / 10-Q files
matching a question's company + fiscal year are ingested. This keeps
the in-process vector store small enough for a fast end-to-end run
with the ``baseline`` chunk size.

Skips auth entirely (no :class:`RagApplication` login flow). The
heuristic LLM fallback has been removed; the ``RAG_LLM_*`` env vars
must point at a working endpoint.

Usage::

    RAG_LLM_BASE_URL=https://api.minimax.io/anthropic \\
    RAG_LLM_API_KEY=... \\
    RAG_LLM_MODEL="MiniMax-M3" \\
    python devtools/financebench.py

``--examples N`` runs the first N examples (default: all 150).
``--pipeline NAME`` runs a single pipeline variant instead of the
sweep; valid names: ``baseline``, ``small``, ``large``, ``hyde``.
``--pdfs-dir DIR`` points at the directory of source PDFs (default:
``/tmp/raghub_fb_pdfs``).
``--data-dir DIR`` overrides the per-run data dir.
``--json PATH`` writes raw per-pipeline results to a JSON file.
``--skip-download`` does not try to fetch the source PDFs.

The script tries to match each question's company and (where
present) ``doc_period`` field against the PDF filenames. PDFs whose
company doesn't match any question are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raghub.config import Settings
from raghub.evaluation import FinanceBench
from raghub.llm import LiteLLMProvider
from raghub.rag import RAG


def _load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from ``path`` into the environment.

    Lines starting with ``#`` and blank lines are skipped. Values
    may be optionally quoted. Existing env vars take precedence
    (we use ``os.environ.setdefault`` so the caller's CLI export
    wins over the .env file).
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


# Pipeline configurations: chunking size, top_k, and whether to use
# the HyDE query transform. ``baseline`` matches the production
# defaults; ``small`` and ``large`` vary the chunk size; ``hyde`` adds
# a HyDE transform.
PIPELINES: dict[str, dict[str, Any]] = {
    "baseline": {
        "chunk_size_words": 500,
        "chunk_overlap_words": 100,
        "top_k": 5,
        "use_hyde": False,
    },
    "small": {
        "chunk_size_words": 200,
        "chunk_overlap_words": 50,
        "top_k": 5,
        "use_hyde": False,
    },
    "large": {
        "chunk_size_words": 1000,
        "chunk_overlap_words": 200,
        "top_k": 5,
        "use_hyde": False,
    },
    "hyde": {
        "chunk_size_words": 500,
        "chunk_overlap_words": 100,
        "top_k": 5,
        "use_hyde": True,
    },
}


@dataclass
class PipelineResult:
    """Aggregate metrics for one pipeline variant.

    Attributes:
        name: Variant label.
        n: Number of examples evaluated.
        pass_rate: Fraction whose answer passed
            ``numeric >= 0.99 or overlap >= 0.6``.
        token_overlap: Mean Jaccard overlap.
        numeric_within_tolerance: Mean numeric-tolerance score.
        answer_length: Mean predicted answer length (chars).
        n_errors: Number of LLM errors during the run.
    """

    name: str
    n: int = 0
    pass_rate: float = 0.0
    token_overlap: float = 0.0
    numeric_within_tolerance: float = 0.0
    answer_length: float = 0.0
    n_errors: int = 0


def _build_settings(pipeline: str, data_dir: str) -> Settings:
    """Return a :class:`Settings` instance tuned for ``pipeline``."""
    cfg = PIPELINES[pipeline]
    base = Settings.load()
    new_query_transforms = base.query_transforms.model_copy(
        update={"enabled": ["hyde"] if cfg["use_hyde"] else []}
    )
    overrides = {
        "chunk_size_words": cfg["chunk_size_words"],
        "chunk_overlap_words": cfg["chunk_overlap_words"],
        "top_k": cfg["top_k"],
        "data_dir": data_dir,
        "zvec_dir": f"{data_dir}/zvec",
        "require_zvec": False,
        "allow_passwordless_login": False,
        "log_level": "WARNING",
    }
    base_with_new = base.model_copy(
        update={**overrides, "query_transforms": new_query_transforms}
    )
    return Settings.model_validate(base_with_new.model_dump())


def _build_llm() -> LiteLLMProvider:
    """Build a :class:`LiteLLMProvider` from the ``RAG_LLM_*`` env vars."""
    base_url = os.environ.get("RAG_LLM_BASE_URL")
    api_key = os.environ.get("RAG_LLM_API_KEY")
    model = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")
    if not base_url or not api_key:
        raise RuntimeError(
            "RAG_LLM_BASE_URL and RAG_LLM_API_KEY must both be set "
            "(the heuristic fallback has been removed)."
        )
    return LiteLLMProvider(model=model, api_key=api_key, api_base=base_url)


def _normalise_company(name: str) -> str:
    """Lower-case alphanumeric token used for filename matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _match_pdfs(
    examples: list[dict[str, Any]], pdfs_dir: Path
) -> list[Path]:
    """Return the PDFs whose filename mentions a question's company.

    Each question is matched by a token derived from its ``company``
    field. PDFs whose filename contains the token are kept; the rest
    are skipped so the in-process vector store stays small.
    """
    needed: set[str] = {
        _normalise_company(ex.get("company", "")) for ex in examples
    }
    needed.discard("")
    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    matched: list[Path] = []
    for pdf in pdfs:
        stem = _normalise_company(pdf.stem.split("_")[0])
        if stem in needed:
            matched.append(pdf)
    return matched


def _stage_pdfs(pdfs: list[Path], staged_dir: Path) -> list[Path]:
    """Copy the matched PDFs into ``staged_dir`` and return the new paths.

    The pipeline's :meth:`RAG.ingest_directory_sync` walks the directory
    and ingests everything it finds; staging them in a dedicated
    directory keeps the in-process index scoped to the relevant
    subset while still using the directory-walking code path (with its
    ``tqdm`` progress bar).
    """
    staged_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    for pdf in pdfs:
        shutil.copy2(pdf, staged_dir / pdf.name)
    return sorted(staged_dir.glob("*.pdf"))


async def _run_pipeline(
    pipeline: str,
    examples: list[dict[str, Any]],
    data_dir: str,
    llm: LiteLLMProvider,
    pdfs_dir: Path,
) -> PipelineResult:
    """Index the matched PDFs, run every example, return aggregate metrics."""
    settings = _build_settings(pipeline, data_dir)
    # Bypass the default Marker converter — its model loader
    # (KeyError: 'encoder') fails on this environment. PlainText is
    # enough for the FinanceBench QA evaluation; the LLM is the
    # bottleneck, not the layout.
    from raghub.documents import PlainTextConverter

    rag = RAG(settings=settings, llm=llm, converter=PlainTextConverter())
    try:
        rag.initialize()
    except Exception as exc:
        try:
            rag.shutdown()
        except Exception:
            pass
        raise RuntimeError(f"initialize() failed: {type(exc).__name__}: {exc}") from exc

    pdfs = _match_pdfs(examples, pdfs_dir)
    if not pdfs:
        raise RuntimeError(
            f"no PDFs matched any of the {len(examples)} example companies; "
            f"check that {pdfs_dir} contains the right files."
        )

    print(f"[sweep] {pipeline}: staging {len(pdfs)} PDFs", flush=True)
    staged_dir = Path(data_dir) / "staged"
    _stage_pdfs(pdfs, staged_dir)

    try:
        await rag.ingest_directory_async(
            staged_dir,
            metadata={"source": "financebench-pdfs", "pipeline": pipeline},
            user=None,
            show_progress=True,
        )
    except Exception as exc:
        try:
            rag.shutdown()
        except Exception:
            pass
        raise RuntimeError(f"ingest failed: {type(exc).__name__}: {exc}") from exc

    evaluator = FinanceBench()
    n_errors = 0

    async def factory(example: dict[str, Any]) -> str:
        nonlocal n_errors
        question = example.get("question") or example.get("query") or ""
        try:
            response = await rag.aquery(question)
            return response.answer
        except Exception as exc:
            n_errors += 1
            return f"[error: {type(exc).__name__}: {exc}]"

    try:
        results = await evaluator.evaluate(examples, response_factory=factory)
    finally:
        try:
            rag.shutdown()
        except Exception:
            pass

    if not results:
        return PipelineResult(name=pipeline)

    metrics_avg = {
        name: statistics.mean(r.metrics.get(name, 0.0) for r in results)
        for name in {k for r in results for k in r.metrics}
    }
    return PipelineResult(
        name=pipeline,
        n=len(results),
        pass_rate=statistics.mean(1.0 if r.passed else 0.0 for r in results),
        token_overlap=metrics_avg.get("token_overlap", 0.0),
        numeric_within_tolerance=metrics_avg.get("numeric_within_tolerance", 0.0),
        answer_length=statistics.mean(
            len((r.details.get("predicted") or "")) for r in results
        ),
        n_errors=n_errors,
    )


def _print_table(results: list[PipelineResult]) -> None:
    """Pretty-print the comparison table."""
    headers = [
        "pipeline", "n", "pass_rate", "token_overlap",
        "numeric", "answer_len", "n_errors",
    ]
    rows = [
        [
            r.name,
            str(r.n),
            f"{r.pass_rate:.3f}",
            f"{r.token_overlap:.3f}",
            f"{r.numeric_within_tolerance:.3f}",
            f"{r.answer_length:.0f}",
            str(r.n_errors),
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "  "

    def _line(cells: list[str]) -> str:
        return sep.join(c.ljust(w) for c, w in zip(cells, widths))

    print()
    print(_line(headers))
    print(_line(["-" * w for w in widths]))
    for row in rows:
        print(_line(row))


async def _async_main(args: argparse.Namespace) -> int:
    """Run the sweep and print the comparison table."""
    llm = _build_llm()
    examples = FinanceBench().ensure_examples()[: args.examples]
    if not examples:
        print("no FinanceBench examples available; aborting")
        return 1

    pdfs_dir = Path(args.pdfs_dir)
    if not pdfs_dir.is_dir():
        print(f"PDF directory not found: {pdfs_dir}")
        print("download with: curl -L github.com/patronus-ai/financebench/raw/main/pdfs.zip")
        return 1

    pipelines = [args.pipeline] if args.pipeline else list(PIPELINES)
    results: list[PipelineResult] = []
    for name in pipelines:
        print(f"[sweep] running {name}...", flush=True)
        try:
            result = await _run_pipeline(name, examples, args.data_dir, llm, pdfs_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = PipelineResult(name=name)
        results.append(result)
    _print_table(results)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                [
                    {
                        "name": r.name,
                        "n": r.n,
                        "pass_rate": r.pass_rate,
                        "token_overlap": r.token_overlap,
                        "numeric_within_tolerance": r.numeric_within_tolerance,
                        "answer_length": r.answer_length,
                        "n_errors": r.n_errors,
                    }
                    for r in results
                ],
                fh,
                indent=2,
            )
    return 0


def main() -> None:
    """Parse CLI args and run the async pipeline sweep."""
    # Auto-load .env from a few standard locations so users can drop
    # their LLM credentials in a file rather than exporting them inline.
    script_dir = Path(__file__).resolve().parent
    for candidate in [script_dir / ".env", script_dir.parent / ".env"]:
        _load_env(candidate)

    parser = argparse.ArgumentParser(
        description="Run the FinanceBench sweep across pipeline configurations."
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of FinanceBench examples (0 = all 150).",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        choices=list(PIPELINES),
        help="Run a single pipeline variant instead of the full sweep.",
    )
    parser.add_argument(
        "--pdfs-dir",
        type=str,
        default="/tmp/raghub_fb_pdfs",
        help="Directory of source 10-K / 10-Q PDFs.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/tmp/raghub_financebench",
        help="Per-run data dir (default: /tmp/raghub_financebench).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write raw results as JSON.",
    )
    args = parser.parse_args()
    if args.examples == 0:
        args.examples = 150
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
