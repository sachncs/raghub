/**
 * Minimal command runner.
 *
 * Built-in argv parser — no commander/yargs dependency. Subcommands
 * register themselves with a name + a description + a runner that
 * receives positional args + flags.
 *
 * `--help` (or `-h`) prints the help table; unknown commands exit
 * non-zero with a hint.
 */

export interface Command {
  readonly name: string;
  readonly description: string;
  readonly usage?: string;
  readonly run: (ctx: CommandContext) => Promise<number>;
}

export interface CommandContext {
  readonly args: readonly string[];
  readonly flags: Readonly<Record<string, string | boolean>>;
  readonly cwd: string;
  readonly env: Readonly<Record<string, string | undefined>>;
}

export const parseArgs = (argv: readonly string[]): {
  args: string[];
  flags: Record<string, string | boolean>;
} => {
  const args: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === undefined) continue;
    if (a.startsWith('--')) {
      const body = a.slice(2);
      const eq = body.indexOf('=');
      if (eq >= 0) {
        const key = body.slice(0, eq);
        const value = body.slice(eq + 1);
        flags[key] = value;
      } else if (i + 1 < argv.length && !(argv[i + 1] ?? '').startsWith('--')) {
        const next = argv[i + 1];
        if (next !== undefined) {
          flags[body] = next;
          i++;
        }
      } else {
        flags[body] = true;
      }
    } else if (a.startsWith('-') && a.length === 2) {
      const key = a.slice(1);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('-')) {
        flags[key] = next;
        i++;
      } else {
        flags[key] = true;
      }
    } else {
      args.push(a);
    }
  }
  return { args, flags };
};

export const printHelp = (commands: readonly Command[]): void => {
  console.log('raghub — multi-user RAG on Strands Agents\n');
  console.log('Usage: raghub <command> [flags]\n');
  console.log('Commands:');
  for (const c of commands) {
    console.log(`  ${c.name.padEnd(28)} ${c.description}`);
  }
  console.log('\nFlags: --help, -h');
};

export const runCommand = async (
  commands: readonly Command[],
  argv: readonly string[],
  env: Readonly<Record<string, string | undefined>>,
  cwd: string,
): Promise<number> => {
  if (argv.length === 0 || argv[0] === '--help' || argv[0] === '-h') {
    printHelp(commands);
    return 0;
  }
  const name = argv[0];
  if (!name) return 0;
  const cmd = commands.find((c) => c.name === name);
  if (!cmd) {
    console.error(`unknown command: ${name}`);
    printHelp(commands);
    return 2;
  }
  const { args, flags } = parseArgs(argv.slice(1));
  if (flags['help'] === true || flags['h'] === true) {
    console.log(`${cmd.name} — ${cmd.description}`);
    if (cmd.usage) console.log(`Usage: ${cmd.usage}`);
    return 0;
  }
  return cmd.run({ args, flags, cwd, env });
};