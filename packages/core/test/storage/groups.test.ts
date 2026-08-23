import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { openWorkspace, type WorkspaceHandle } from '../../src/workspace.js';
import { SqliteRoleStore, SqliteGroupStore } from '../../src/storage/groups.js';

const PATH = ':memory:';

describe('SqliteRoleStore + SqliteGroupStore', () => {
  let handle: WorkspaceHandle;
  let roles: SqliteRoleStore;
  let groups: SqliteGroupStore;

  beforeEach(async () => {
    handle = await openWorkspace({ path: PATH });
    roles = new SqliteRoleStore({ db: handle.db });
    groups = new SqliteGroupStore({ db: handle.db });
  });

  afterEach(async () => {
    await roles.close();
    await groups.close();
    handle.close();
  });

  it('creates and reads back a role', async () => {
    const r = await roles.create({ name: 'analyst', description: 'reads data' });
    expect(r.name).toBe('analyst');
    const got = await roles.get(r.id);
    expect(got?.name).toBe('analyst');
  });

  it('adds and removes role members (users and groups)', async () => {
    const r = await roles.create({ name: 'engineer' });
    await roles.addMember(r.id, 'user', 'usr_alice');
    await roles.addMember(r.id, 'group', 'grp_team_a');
    const members = await roles.members(r.id);
    expect(members.length).toBe(2);
    await roles.removeMember(r.id, 'user', 'usr_alice');
    expect((await roles.members(r.id)).length).toBe(1);
  });

  it('creates a group and resolves user groups', async () => {
    const g = await groups.create({ name: 'eng' });
    await groups.addMember(g.id, 'usr_1');
    await groups.addMember(g.id, 'usr_2');
    const userGroups = await groups.userGroups('usr_1');
    expect(userGroups.length).toBe(1);
    expect(userGroups[0]?.name).toBe('eng');
  });

  it('lists groups and roles in alphabetical order', async () => {
    await roles.create({ name: 'zeta' });
    await roles.create({ name: 'alpha' });
    const r = await roles.list();
    expect(r.map((x) => x.name)).toEqual(['alpha', 'zeta']);

    await groups.create({ name: 'two' });
    await groups.create({ name: 'one' });
    const g = await groups.list();
    expect(g.map((x) => x.name)).toEqual(['one', 'two']);
  });
});
