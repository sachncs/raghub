'use client';

import { useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export default function OnboardingPage() {
  const [tenantName, setTenantName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const res = await fetch('/api/proxy', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-raghub-path': '/v1/auth/register' },
      body: JSON.stringify({ tenantName, email, password }),
    });
    const body = await res.json().catch(() => ({}));
    setBusy(false);
    if (!res.ok) {
      setError(body?.error?.message ?? 'register failed');
      return;
    }
    document.cookie = `raghub_token=${body.token}; path=/; max-age=86400; samesite=lax`;
    window.location.href = '/chat';
  };

  return (
    <main className="container max-w-md py-12">
      <h1 className="mb-1 text-2xl font-semibold">Create your workspace</h1>
      <p className="mb-6 text-sm text-muted-foreground">
        Phase 1: minimal registration. The full onboarding wizard (mode, hybrid weights, multimodal, traces) lands in Phase 2.
      </p>
      <form onSubmit={submit} className="space-y-4 rounded-lg border bg-card p-6 text-card-foreground">
        <div className="space-y-2">
          <Label htmlFor="tenant">Workspace name</Label>
          <Input id="tenant" value={tenantName} onChange={(e) => setTenantName(e.target.value)} required />
        </div>
        <div className="space-y-2">
          <Label htmlFor="email">Admin email</Label>
          <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input id="password" type="password" minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Button type="submit" disabled={busy} className="w-full">{busy ? 'Creating…' : 'Create workspace'}</Button>
      </form>
    </main>
  );
}