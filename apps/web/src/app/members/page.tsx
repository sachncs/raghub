'use client';

import { useEffect, useState } from 'react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

interface Member {
  userId: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  joinedAt: string;
}

const proxy = async (
  path: string,
  init: RequestInit = {},
): Promise<Response> =>
  fetch('/api/proxy', {
    ...init,
    headers: { ...init.headers, 'x-raghub-path': path },
  });

const roleBadge = (role: Member['role']) => {
  const variant =
    role === 'owner'
      ? 'default'
      : role === 'admin'
        ? 'secondary'
        : 'outline';
  return <Badge variant={variant}>{role}</Badge>;
};

export default function MembersPage() {
  const [members, setMembers] = useState<Member[]>([]);
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<'admin' | 'member' | 'viewer'>('member');
  const [error, setError] = useState<string | null>(null);
  const [inviting, setInviting] = useState(false);

  const refresh = async (): Promise<void> => {
    const res = await proxy('/v1/workspaces/members');
    if (!res.ok) return;
    const body = (await res.json()) as { members: Member[] };
    setMembers(body.members);
  };

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
      return;
    }
    void refresh();
  }, []);

  const invite = async (): Promise<void> => {
    setInviting(true);
    setError(null);
    try {
      const res = await proxy('/v1/workspaces/members', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email, role }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const message = (body as { error?: { message?: string } })?.error?.message ?? 'invite failed';
        setError(message);
        toast.error(message);
        return;
      }
      toast.success('Invited');
      setEmail('');
      await refresh();
    } finally {
      setInviting(false);
    }
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
    <main className="container max-w-3xl py-8">
      <h1 className="mb-6 text-2xl font-semibold">Members</h1>

      <Dialog>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invite a member</DialogTitle>
            <DialogDescription>
              They will be added with the chosen role.
            </DialogDescription>
          </DialogHeader>
          <div className="flex items-center gap-2">
            <Label htmlFor="email" className="sr-only">
              Email
            </Label>
            <Input
              id="email"
              type="email"
              placeholder="user@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="flex-1"
            />
            <Select value={role} onValueChange={(v) => setRole(v as typeof role)}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="admin">admin</SelectItem>
                <SelectItem value="member">member</SelectItem>
                <SelectItem value="viewer">viewer</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={invite} disabled={inviting || email.length === 0}>
              {inviting ? 'Inviting…' : 'Invite'}
            </Button>
          </div>
          {error && (
            <p className="mt-2 text-sm text-destructive" role="alert">
              {error}
            </p>
          )}
        </DialogContent>
      </Dialog>

      {members.length === 0 ? (
        <p className="text-sm text-muted-foreground">No members yet.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Role</TableHead>
              <TableHead className="text-right">Joined</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {members.map((m) => (
              <TableRow key={m.userId}>
                <TableCell className="font-mono text-xs">{m.userId}</TableCell>
                <TableCell>{roleBadge(m.role)}</TableCell>
                <TableCell className="text-right text-xs text-muted-foreground">
                  {new Date(m.joinedAt).toLocaleDateString()}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex items-center justify-end gap-2">
                    <Select
                      value={m.role}
                      onValueChange={(v) => void changeRole(m.userId, v as Member['role'])}
                    >
                      <SelectTrigger className="h-8 w-[120px] text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="owner">owner</SelectItem>
                        <SelectItem value="admin">admin</SelectItem>
                        <SelectItem value="member">member</SelectItem>
                        <SelectItem value="viewer">viewer</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void remove(m.userId)}
                    >
                      Remove
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      <DialogFooter className="mt-6">
        <p className="text-xs text-muted-foreground">
          Only admins and owners can change roles.
        </p>
      </DialogFooter>
    </main>
  );
}