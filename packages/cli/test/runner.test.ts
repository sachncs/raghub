import { describe, expect, it } from 'vitest';

import { parseArgs, printHelp, runCommand } from '../src/runner.js';

describe('parseArgs', () => {
  it('parses --flag value pairs', () => {
    const { args, flags } = parseArgs(['--port', '3000', '--verbose']);
    expect(args).toEqual([]);
    expect(flags['port']).toBe('3000');
    expect(flags['verbose']).toBe(true);
  });

  it('parses --flag=value', () => {
    const { flags } = parseArgs(['--port=3000']);
    expect(flags['port']).toBe('3000');
  });

  it('parses short -f value', () => {
    const { flags } = parseArgs(['-p', '3000']);
    expect(flags['p']).toBe('3000');
  });

  it('treats positional args separately from flags', () => {
    const { args, flags } = parseArgs(['init', '--port', '3000']);
    expect(args).toEqual(['init']);
    expect(flags['port']).toBe('3000');
  });
});

describe('runCommand', () => {
  const cmd = {
    name: 'ping',
    description: 'echo ping',
    async run({ flags }: { flags: Readonly<Record<string, string | boolean>> }) {
      return Number(flags['code'] ?? 0);
    },
  };

  it('returns 0 on success', async () => {
    const code = await runCommand([cmd], ['ping'], {}, '/tmp');
    expect(code).toBe(0);
  });

  it('prints help on --help', async () => {
    const code = await runCommand([cmd], ['--help'], {}, '/tmp');
    expect(code).toBe(0);
  });

  it('rejects unknown commands with exit 2', async () => {
    const code = await runCommand([cmd], ['nope'], {}, '/tmp');
    expect(code).toBe(2);
  });

  it('prints help table without throwing', () => {
    expect(() => printHelp([cmd])).not.toThrow();
  });
});