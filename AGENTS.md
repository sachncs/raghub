# Coding Standards — raghub (TypeScript)

These are the rules every contribution must follow. They mirror the spirit of
the original Python `AGENTS.md` (now archived) and are tuned for the
TypeScript monorepo on Strands Agents.

## Layout

```text
raghub/
├── package.json                # pnpm workspace root
├── pnpm-workspace.yaml
├── tsconfig.base.json          # strict, exactOptionalPropertyTypes
├── turbo.json
├── AGENTS.md                   # this file
├── packages/
│   ├── core/                   # @raghub/core — always installed
│   ├── orchestrator/           # @raghub/orchestrator — Strands wrapper
│   ├── api/                    # @raghub/api — Hono HTTP layer
│   ├── cli/                    # @raghub/cli — `raghub` binary
│   └── eval/                   # @raghub/eval — Finance + Frames + CARE
├── apps/
│   ├── web/                    # Next.js 15 + shadcn/ui
│   └── docs/                   # Astro
└── archive/                    # frozen Python 0.9.x release, read-only
```

Each package owns its `src/`, `test/`, `dist/`, `package.json`, `tsconfig.json`.
Workspace dependencies use the workspace protocol (`"@raghub/core":
"workspace:*"`).

## TypeScript style

- **TS 5.6+**, ESM only (`"type": "module"`), Node 22+ LTS.
- **Strict mode** is the baseline: `strict`, `noImplicitOverride`,
  `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`,
  `noFallthroughCasesInSwitch`, `noImplicitReturns`.
- **No `any`**. Use `unknown` plus a type guard, or a precise type. If you
  really cannot type something, write the smallest possible interface and
  cast with `as` at the boundary; never sprinkle `any` through a module.
- **Branded types** for IDs: `type UserId = string & { readonly __brand:
  'UserId' }`. No bare `string` IDs in public signatures.
- **Frozen classes** for value objects (the TS equivalent of the Python
  `@dataclass(slots=True, frozen=True)`). Use `Object.freeze` in the
  constructor; fields are `readonly`.
- **No `enum`**. Use `const X = { Foo: 'foo' } as const; type X = typeof
  X[keyof typeof X]`.
- **`as const` for literal objects**, especially discriminated unions.
- **No default exports** in library code. Named exports only.
- **Exports are explicit**: every package has a top-level `index.ts` that
  re-exports the public surface. Use the `export * from './foo'` form only
  for namespaces; otherwise prefer named re-exports.
- **Errors are classes**. Throw `new FooError(...)`. The framework has a
  `RaghubError` base; every domain extends it (auth, ingest, retrieval,
  generation, vector store, verification, configuration, missing dep).
- **Reserved names are forbidden**: `Object`, `Function`, `Promise`,
  `String`, `Number`, `Boolean`, `Error`, `Type`, `default`, `next`. The
  Python AGENTS rule applies verbatim.
- **Shadowing built-ins** is forbidden: `id`, `name`, `length`, `value`,
  `type`, `parent`, `data`, `next`, `prev`, `open`, `close`. Use the
  `entity_` prefix only where unavoidable; otherwise rename.

## Modules and dependency hygiene

- **Explicit dependencies** in `package.json`. Do not rely on transitive
  imports; if you `import` it, you must list it.
- **External dep bounds**: stable (`>=1.0.0`) → `>=<known-good>,<next-major>`;
  prerelease (`dev`/`a`/`b`/`rc`) → bounded within the same line; `<1.0.0`
  → narrow range validated by tests.
- **No `**/dist/**` cross-package imports**. Always import from the
  workspace source root via the workspace protocol.
- **No barrel re-exports** in deeply-nested paths. A `src/index.ts` per
  package, no intermediate `index.ts` files.

## Async and concurrency

- The library is **async-first**. If a function can suspend, it must
  return `Promise<T>`. No sync wrappers around async I/O.
- Use **`AbortSignal`** for cancellation. Every public async function
  accepts `signal?: AbortSignal`.
- **`for await ... of`** for streams. Do not roll your own async iterator.

## Testing

- **Vitest** for unit and integration. No Jest.
- Tests live under `test/` (not `tests/`), mirroring `src/` layout.
- One file per concept: `test/retrieval/hybrid.test.ts` covers
  `src/retrieval/hybrid.ts`. No mega-test files.
- Use **`describe`/`it`**; never `test` alone.
- Property tests with **`fast-check`**, mirrored from the Python
  Hypothesis suite where applicable.
- Coverage target: ≥ 80% per package. CI fails below.
- **One assertion concept per `it`**. Multi-assertion tests must use
  `expect(...).toMatchInlineSnapshot()` or split into separate tests.

## Public API surface

- Every package's `src/index.ts` defines `export *` of the curated public
  API, **plus** an `__internal` namespace for items consumed by other
  packages but not by users.
- Breaking changes require a major version bump. The `CHANGELOG.md` entry
  must list the migration path.

## Performance

- **Cache expensive computations** behind a lazy getter; invalidate on
  mutation, not eagerly.
- **Avoid `JSON.parse(JSON.stringify(x))`** for clone. Use `structuredClone`
  or a domain-aware copy.
- **Async in parallel** with `Promise.all` / `Promise.allSettled`. Never
  sequentially `await` inside a loop unless ordering is required.

## Lint and format

- **Prettier** is authoritative. No opinions.
- **`eslint`** with `@typescript-eslint`, `eslint-plugin-import`,
  `eslint-plugin-simple-import-sort`, `eslint-plugin-unicorn`. Run via
  `pnpm lint`.
- **`oxc`** is the fast path; `eslint` is the slow path. CI runs both.
- CI runs `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build`.

## Git and commits

- **Atomic commits**, one logical change per commit.
- Conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `test:`,
  `docs:`, `chore:`, `perf:`.
- Branch names: `feat/<short>`, `fix/<short>`, `refactor/<short>`.
- PR titles match the commit subject. PR body fills the migration notes
  section when the change is breaking.

## Deprecation

When renaming or removing a public symbol:

1. Mark the old name `@deprecated` in JSDoc and re-export pointing to the
   new name.
2. Emit a `console.warn` on first use, gated by a process-global flag so
   tests can suppress it.
3. Remove the alias two minor versions later.
4. Update the migration doc.
