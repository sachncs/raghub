/**
 * Runtime helpers for the CLI.
 *
 * The CLI's `server` subcommand delegates to `@revex/api/start`
 * for the actual composition. This module exists so the CLI can
 * wrap that start function with version banner, signal handling,
 * and structured logging.
 */

export interface RuntimeOptions {
  readonly port: number;
  readonly host: string;
  readonly autostartWorker: boolean;
}

export async function start(opts: RuntimeOptions): Promise<void> {
  const { start: apiStart } = await import('@revex/api');
  void opts;
  await apiStart();
}