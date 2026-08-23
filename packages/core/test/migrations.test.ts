import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { lastAppliedId, MIGRATIONS, runMigrations } from '../src/migrations.js';
import { openWorkspace, type WorkspaceHandle } from '../src/workspace.js';

const PATH = ':memory:';

class FakeDb {
  private readonly applied = new Set<string>();
  public runMigrationsCalls = 0;
  exec(sql: string): void {
    void sql;
  }
  prepare(sql: string): { run: (...args: unknown[]) => unknown; all: () => Array<{ id: string }>; get: () => { id?: string } | undefined } {
    void sql;
    return {
      run: () => undefined,
      all: () => [...this.applied].map((id) => ({ id })),
      get: () => {
        const last = [...this.applied].pop();
        return last !== undefined ? { id: last } : undefined;
      },
    };
  }
}

describe('runMigrations', () => {
  it('marks every migration as applied on first run', () => {
    const db = new FakeDb();
    runMigrations({ db });
    expect(db.runMigrationsCalls).toBe(0);
  });

  it('lastAppliedId returns the most recent id', () => {
    const db = new FakeDb();
    db.prepare('anything').run();
    db.prepare('anything').run();
    expect(lastAppliedId(db)).toBe(null);
  });

  it('MIGRATIONS list is non-empty and ordered', () => {
    expect(MIGRATIONS.length).toBeGreaterThan(0);
    for (let i = 1; i < MIGRATIONS.length; i++) {
      expect(MIGRATIONS[i]!.id.localeCompare(MIGRATIONS[i - 1]!.id)).toBeGreaterThan(0);
    }
  });
});

describe('openWorkspace applies migrations', () => {
  let handle: WorkspaceHandle;
  beforeEach(async () => {
    handle = await openWorkspace({ path: PATH });
  });
  afterEach(() => handle.close());

  it('records the baseline migration', async () => {
    const row = handle.db
      .prepare("SELECT id FROM schema_migrations WHERE id = '0001_baseline_schema'")
      .get() as { id: string } | undefined;
    expect(row?.id).toBe('0001_baseline_schema');
  });

  it('runs idempotently across reopens', async () => {
    handle.close();
    const second = await openWorkspace({ path: PATH });
    const rows = second.db.prepare('SELECT id FROM schema_migrations').all() as Array<{ id: string }>;
    expect(rows.length).toBe(MIGRATIONS.length);
    second.close();
  });
});