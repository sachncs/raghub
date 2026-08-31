import { describe, expect, it } from 'vitest';

import { buildProgram, VERSION } from '../src/index.js';

describe('@revex/cli', () => {
  it('exports the right version', () => {
    expect(VERSION).toBe('1.1.0');
  });

  it('builds a commander program with the expected commands', () => {
    const program = buildProgram();
    const names = program.commands.map((c) => c.name());
    expect(names).toContain('init');
    expect(names).toContain('server');
    expect(names).toContain('ingest');
    expect(names).toContain('query');
    expect(names).toContain('feedback');
    expect(names).toContain('config');
    expect(names).toContain('tenant');
    expect(names).toContain('backup');
    expect(names).toContain('queue');
    expect(names).toContain('migrate');
    expect(names).toContain('eval');
  });
});