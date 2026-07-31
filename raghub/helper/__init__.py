"""raghub.helper namespace package.

This module is staged explicitly so the package is no longer an
implicit namespace (PEP 420). Phase 2 collapses ``helper`` into
``raghub.api_auth`` / ``raghub.cli_commands`` / ``raghub.api_*``;
this ``__init__`` then goes away together with the rest of the
package.

Until Phase 2 lands, ``helper`` continues to expose the FastAPI
deps, the response helpers, the rate-limiter middleware, the SSE
helpers, the CLI command classes, and the search-dispatch
helpers — re-imported here from their respective submodules for
backwards-compatible ``from raghub.helper import X`` access.

The re-exports here are deliberately limited to the symbols that
the rest of the package (and devtools) need. Adding a symbol to
this file is fine; adding a new sub-package is not — that path
goes through Phase 2 instead.
"""
