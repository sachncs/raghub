"""RAGHub CLI.

A small `argparse`-based CLI that exposes the most common framework
operations:

* ``raghub init`` — emit a sample ``raghub.yaml`` to stdout.
* ``raghub ingest PATH`` — convert + chunk + embed + index a file.
* ``raghub query "..."`` — ask a question.
* ``raghub eval financebench`` — run the FinanceBench benchmark.
* ``raghub health`` — liveness probe.
* ``raghub version`` — print the package version.

All commands route through the public :class:`raghub.RAG` facade.
"""
