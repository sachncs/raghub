/**
 * Graceful shutdown coordinator.
 *
 * Tracks async cleanup handlers (workspace pool drain, telemetry
 * flush, audit flush) and runs them in registration order on
 * SIGTERM/SIGINT. Idempotent: subsequent calls are no-ops.
 */

export type CleanupHandler = () => Promise<void>;

export class ShutdownCoordinator {
  private readonly handlers: { readonly name: string; readonly handler: CleanupHandler }[] = [];
  private invoked = false;

  register(name: string, handler: CleanupHandler): void {
    this.handlers.push({ name, handler });
  }

  async run(reason: string): Promise<void> {
    if (this.invoked) return;
    this.invoked = true;
    process.stdout.write(`revex-api: ${reason}; running ${this.handlers.length} cleanup handlers\n`);
    for (const h of this.handlers) {
      try {
        await h.handler();
        process.stdout.write(`revex-api: ✓ ${h.name}\n`);
      } catch (err) {
        process.stderr.write(`revex-api: ✗ ${h.name}: ${err instanceof Error ? err.message : String(err)}\n`);
      }
    }
  }

  attachSignalHandlers(reason = 'shutdown requested'): void {
    const signal = (sig: NodeJS.Signals): void => {
      void this.run(`${sig} (${reason})`).then(() => process.exit(0));
    };
    process.once('SIGTERM', signal);
    process.once('SIGINT', signal);
  }

  count(): number {
    return this.handlers.length;
  }
}