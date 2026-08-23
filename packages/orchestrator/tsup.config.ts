import { defineConfig } from 'tsup';

export default defineConfig({
  entry: { index: 'src/index.ts', __internal: 'src/__internal.ts' },
  format: ['esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  target: 'node22',
  external: ['strands-agents'],
});