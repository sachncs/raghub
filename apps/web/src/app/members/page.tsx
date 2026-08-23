'use client';

import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface Member {
  userId: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  joinedAt: string;
}

const proxy = async (
  path: string,
  init: RequestInit = {},
): Promise<Response> => {
  return fetch('/api/proxy', {
    ...init,
    headers: { ...init.headers, 'x-raghub-path': path },
  });
};

export default function MembersPage() {
  const [members, setMembers] = useState<Member[]>([]);
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<'admin' | 'member' | 'viewer'>('member');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
      return;
    }
    void refresh();
  }, []);

  const refresh = async (): Promise<void> => {
    const res = await proxy('/v1/workspaces/members');
    if (!res.ok) return;
    const body = (await res.json()) as { members: Member[] };
    setMembers(body.members);
  };

  const invite = async (): Promise<void> => {
    setError(null);
    const res = await proxy('/v1/workspaces/members', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email, role }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setError(body?.error?.message ?? 'invite failed');
      return;
    }
    setEmail('');
    await refresh();
  };

  const changeRole = async (userId: string, next: Member['role']): Promise<void> => {
    await proxy(`/v1/workspaces/members/${userId}`, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ role: next }),
    });
    await refresh();
  };

  const remove = async (userId: string): Promise<void> => {
    await proxy(`/v1/workspaces/members/${userId}`, { method: 'DELETE' });
    await refresh();
  };

  return (
    <main className="container max-w-2xl py-8">
      <h1 className="mb-6 text-2xl font-semibold">Members</h1>
      <div className="mb-6 rounded-lg border bg-card p-4">
        <div className="flex items-center gap-2">
          <Input
            placeholder="user@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="flex-1"
          />
          <select
            className="rounded border bg-background px-2 py-1 text-sm"
            value={role}
            onChange={(e) => setRole(e.target.value as typeof role)}
          >
            <option value="admin">admin</option>
            <option value="member">member</option>
            <option value="viewer">viewer</option>
          </select>
          <Button onClick={invite}>Invite</Button>
        </div>
        {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
      </div>
      <ul className="space-y-1 text-sm">
        {members.map((m) => (
          <li
            key={m.userId}
            className="flex items-center justify-between rounded border px-3 py-2"
          >
            <span>{m.userId}</span>
            <span className="flex items-center gap-2">
              <select
                className="rounded border bg-background px-2 py-1 text-xs"
                value={m.role}
                onChange={(e) => void changeRole(m.userId, e.target.value as Member['role'])}
              >
                <option value="owner">owner</option>
                <option value="admin">admin</option>
                <option value="member">member</option>
                <option value="viewer">viewer</option>
              </select>
              <button
                type="button"
                onClick={() => void remove(m.userId)}
                className="text-xs text-destructive underline"
              >
                remove
              </button>
            </span>
          </li>
        ))}
      </ul>
    </main>
  );
}