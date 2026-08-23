import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { brandId, type UserId, type WorkspaceId } from '../../src/domain/index.js';
import { openWorkspace, type WorkspaceHandle } from '../../src/workspace.js';
import { SqliteAuditEventStore } from '../../src/storage/audit.js';

const PATH = ':memory:';

describe('SqliteAuditEventStore', () => {
  let handle: WorkspaceHandle;
  let audit: SqliteAuditEventStore;
  const wsp = brandId<WorkspaceId>('wsp_1');
  const alice = brandId<UserId>('usr_alice');

  beforeEach(async () => {
    handle = await openWorkspace({ path: PATH });
    audit = new SqliteAuditEventStore({ db: handle.db });
  });

  afterEach(async () => {
    await audit.close();
    handle.close();
  });

  it('records an event with detail', async () => {
    await audit.record({
      kind: 'auth.register',
      workspaceId: wsp,
      actorId: alice,
      resourceId: null,
      detail: { email: 'alice@example.com' },
    });
    const list = await audit.list({ workspaceId: wsp });
    expect(list.length).toBe(1);
    expect(list[0]?.kind).toBe('auth.register');
    expect(list[0]?.detail['email']).toBe('alice@example.com');
  });

  it('filters by workspace', async () => {
    await audit.record({ kind: 'document.upload', workspaceId: wsp, actorId: alice, resourceId: 'doc_1', detail: {} });
    const other = brandId<WorkspaceId>('wsp_2');
    const list = await audit.list({ workspaceId: other });
    expect(list.length).toBe(0);
  });

  it('respects since', async () => {
    await audit.record({ kind: 'ingest.complete', workspaceId: wsp, actorId: alice, resourceId: 'doc_1', detail: {} });
    const future = new Date(Date.now() + 60_000);
    const list = await audit.list({ workspaceId: wsp, since: future });
    expect(list.length).toBe(0);
  });
});