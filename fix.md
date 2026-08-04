# Plan: Remove ragas dependency + HeuristicProvider

## Theme

Drop two non-core pieces of raghub:

1. **`ragas`** — an optional eval adapter that has 4 always-skipped tests and
   pulls in `langchain`, `langchain-community`, `langchain-core`,
   `langchain-text-splitters` as transitive deps (~150 MB).
2. **`HeuristicProvider`** — an offline LLM fallback that extracted the
   highest-overlap sentence from the context. With no LLM API key set,
   raghub now raises `ConfigurationError` instead of returning a degraded
   fallback. Per AGENTS.md R4 ("no shims, no aliases"), drop the public
   `Heuristic` alias too.

## Acceptance criteria

| Rule | Detail |
|---|---|
| R1 | No `# noqa` / `# type: ignore` introduced |
| R2 | No `_`-prefix identifiers |
| R3 | No `Manager`/`Helper`/etc. class name suffixes |
| R4 | No back-compat aliases — `HeuristicProvider` and `Heuristic` are
  gone, no stub re-export |
| R8 | `verify()` invocations unchanged |
| R10 | No new `Any` outside metadata |
| C1 | No new functions > 40 LOC |
| Tests | All 1148 → ~1130 still pass (the 4 skipped ragas tests disappear,
  the 8 HeuristicProvider tests disappear, the 4 security tests get
  rewritten) |

## Files to delete (4)

1. `raghub/eval/ragas/__init__.py` (270 LOC, includes `RagasAdapter`,
   `import_ragas`, `load_metric`, `build_dataset`, `extract_scores`)
2. `raghub/eval/ragas/` (the package directory)
3. `tests/test_heuristic_llm.py` (38 LOC, 8 tests)
4. `tests/test_ragas_adapter.py` (~50 LOC, 4 tests, all skipped)

## Files to modify (8)

### `raghub/llm.py`
```diff
@@ -32,7 +32,6 @@
 __all__ = [
     "Generator",
     "GenerationRequest",
-    "HeuristicProvider",
     "LiteLLM",
     "any_llm_api_key_present",
 ]
@@ -141,62 +140,6 @@ class LiteLLM(Generator):
     return False


-class HeuristicProvider(Generator):
-    """Offline LLM provider that answers from context directly.
-
-    Uses a simple heuristic — extracts the most relevant sentence from the
-    context, or returns a canned response when no context is given.
-    No API key or network access required.
-    """
-
-    model_name: str = "heuristic"
-
-    @staticmethod
-    def generate(request: GenerationRequest) -> str:
-        """Generate an answer from context using simple heuristics."""
-        if not request.context:
-            return "No context was retrieved. Configure an LLM API key for full answer generation."
-        # Normalise: context may be Sequence[str] or Sequence[Hit].
-        texts: list[str] = []
-        for entry in request.context:
-            if isinstance(entry, str):
-                texts.append(entry)
-            else:
-                texts.append(getattr(getattr(entry, "chunk", entry), "text", str(entry)))
-        question_lower = request.question.lower()
-        question_words = set(question_lower.split())
-        scored: list[tuple[int, str]] = []
-        for chunk in texts:
-            for sentence in chunk.split("."):
-                stripped = sentence.strip()
-                if not stripped:
-                    continue
-                lowered = stripped.lower()
-                score = sum(1 for w in question_words if w in lowered)
-                scored.append((score, stripped))
-        scored.sort(key=lambda x: -x[0])
-        if scored:
-            return scored[0][1]
-        return (texts[0] if texts else "")[:500]
-
-
@@ -506,7 +449,7 @@ def build_llm(model_name: str, api_key: str | None = None) -> Generator:
     * No API key available → :class:`HeuristicProvider` (offline).
+    * No API key available → :class:`ConfigurationError` is raised; callers
+      must set an LLM API key explicitly.
     """
     if not any_llm_api_key_present() and not api_key:
-        return HeuristicProvider()
+        raise ConfigurationError(
+            "No LLM API key configured; set one in Settings "
+            "(e.g. RAG_LLM_API_KEY) or pass api_key= explicitly."
+        )
     return LiteLLM(model=model_name, api_key=api_key)
```

### `raghub/__init__.py`
```diff
@@ -90,10 +90,7 @@
 # Inference providers.
-from raghub.llm import (
-    HeuristicProvider as Heuristic,
-)
+from raghub.llm import (
+    LiteLLM,
+)
@@ -107,7 +104,6 @@ __all__ = [
     "Generator",
     "GenerationRequest",
-    "Heuristic",
     "LiteLLM",
     "Manifest",
```

### `raghub/rag.py`
```diff
@@ -247,12 +247,9 @@ def has_llm_api_key() -> bool:
     Returns:
         True when at least one LLM API key env var is set (or a key is
-        explicitly provided via constructor). When False, the offline
-        :class:`HeuristicProvider` is used as the fallback.
+        explicitly provided via constructor). When False, callers must
+        configure an LLM API key before invoking the LLM.
     """
     return any(_is_set(env_var) for env_var in LLM_API_KEY_ENV_VARS)
-
-
-# Keep the legacy alias for callers that imported HeuristicProvider.
-HeuristicProvider = _import_alias()
```

### `raghub/eval/synthetic.py`
```diff
@@ -8,7 +8,7 @@
-    ``HeuristicProvider`` — a simple sentence-level extractor that picks the
+    ``LiteLLMProvider`` (with a configured API key) — for full LLM-backed
     most-overlapping sentence. Use it as a smoke test for the eval
-    pipeline without an API key.
+    pipeline; configure an LLM API key for production evaluation runs.
```

### `pyproject.toml:67`
```diff
- ragas = ["ragas>=0.1,<1"]
```

### `README.md`
- Drop the ragas mention if present (verify with `grep -i ragas README.md`).

### `tests/test_llm.py`
- Drop every `HeuristicProvider`-referring test (16 test methods; keep
  all `LiteLLM` tests).

### `tests/test_security.py`
- Rewrite the 4 tests that depended on the HeuristicProvider fallback
  (lines 40, 61-96). New contract: no LLM key → `ConfigurationError`.

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `HeuristicProvider` was a documented offline fallback | Medium | Documentation gap | Update README + docstrings |
| `build_llm()` may be called by tests without a real key | High | Test breakage | Rewrite the affected tests in `test_security.py` and `test_llm.py` |
| `ragas` was a soft dependency | Low | None | Tests were already skipped |
| `from raghub import Heuristic` import | Medium | Breaking change | Document in CHANGELOG (R4 explicit) |

## Verification

```bash
# 1. Public import still works
python -c "from raghub import RAG; print(RAG)"

# 2. No remaining references
grep -rn "HeuristicProvider\|Heuristic\b\|ragas" raghub/ tests/ pyproject.toml
# Expected output: empty

# 3. build_llm() now raises without a key
RAGHUB_LLM_API_KEY="" python -c "
from raghub.llm import build_llm
try:
    build_llm('gpt-4o-mini')
except Exception as e:
    print(type(e).__name__, e)
"

# 4. All tests pass
python -m pytest tests/ --no-cov -q

# 5. CHANGELOG entry
git commit -m "v0.9.5: remove HeuristicProvider and ragas dependency

Per R4 (no back-compat aliases). Two separate commits for traceability:
HeuristicProvider removal, then ragas removal."
```

## Success criteria

- `HeuristicProvider` class and `Heuristic` public alias are **gone** from the codebase
- `raghub/eval/ragas/` directory is **gone**
- `[ragas]` extra in `pyproject.toml` is **gone**
- No new `Any` annotations outside `metadata`
- No new functions > 40 LOC
- All non-skipped tests pass (~1130 expected; was 1148, with -8 Heuristic tests + -4 skipped ragas tests + -4 rewritten security tests)

## Order of execution (atomic commits for clean bisect)

1. Commit 1: Remove HeuristicProvider + tests
2. Commit 2: Remove ragas + tests + pyproject extra
3. CHANGELOG entry in both commits
