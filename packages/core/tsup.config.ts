import { defineConfig } from 'tsup';

export default defineConfig({
  entry: { index: 'src/index.ts', __internal: 'src/__internal.ts' },
  format: ['esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  target: 'node22',
  splitting: false,
  /* sqlite-vec must stay external: its `load(db)` function
   * uses `import.meta.resolve('sqlite-vec-<platform>-<arch>')`
   * to locate the native .dylib/.so/.dll. tsup would otherwise
   * inline the wrapper, which breaks the platform package
   * resolution. The same applies to its peer packages. */
  external: [
    'sqlite-vec',
    'better-sqlite3',
    /^sqlite-vec-/,
  ],
  noExternal: false,
});