/**
 * Database handle type used by the API server. Matches the
 * better-sqlite3 instance shape.
 */

export interface Database {
  prepare(sql: string): {
    run(...args: unknown[]): unknown;
    get(...args: unknown[]): unknown;
    all(...args: unknown[]): unknown[];
  };
  exec(sql: string): void;
  pragma?(k: string): unknown;
  close(): void;
}