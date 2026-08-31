/**
 * Structured logger.
 *
 * `Logger` mirrors the legacy loguru adapter: bind a name +
 * structured kwargs to every record, emit JSON to stdout/stderr
 * or a custom sink. The default sink is `console.log`; swap via
 * `setSink(fn)`.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogRecord {
  readonly level: LogLevel;
  readonly message: string;
  readonly component: string;
  readonly timestamp: string;
  readonly fields: Readonly<Record<string, unknown>>;
}

export type LogSink = (record: LogRecord) => void;

let sink: LogSink = (record) => {
  process.stdout.write(JSON.stringify(record) + '\n');
};

export const setSink = (next: LogSink): void => {
  sink = next;
};

export const createLogger = (component: string) => {
  const emit = (level: LogLevel, message: string, fields: Readonly<Record<string, unknown>> = {}): void => {
    sink({
      level,
      message,
      component,
      timestamp: new Date().toISOString(),
      fields,
    });
  };
  return {
    debug: (msg: string, fields?: Readonly<Record<string, unknown>>): void => emit('debug', msg, fields),
    info: (msg: string, fields?: Readonly<Record<string, unknown>>): void => emit('info', msg, fields),
    warn: (msg: string, fields?: Readonly<Record<string, unknown>>): void => emit('warn', msg, fields),
    error: (msg: string, fields?: Readonly<Record<string, unknown>>): void => emit('error', msg, fields),
  };
};

export type Logger = ReturnType<typeof createLogger>;