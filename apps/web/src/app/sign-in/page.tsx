'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export default function SignInPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passphrase, setPassphrase] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    const res = await fetch('/api/proxy', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-raghub-path': '/v1/auth/login' },
      body: JSON.stringify({ email, password, passphrase }),
    });
    const body = (await res.json().catch(() => ({}))) as {
      token?: string;
      error?: { message?: string };
    };
    setSubmitting(false);
    if (!res.ok || !body.token) {
      setError(body?.error?.message ?? 'login failed');
      return;
    }
    document.cookie = `raghub_token=${body.token}; path=/; max-age=86400; samesite=lax`;
    if (passphrase.length > 0) {
      document.cookie = `raghub_passphrase=${encodeURIComponent(passphrase)}; path=/; max-age=86400; samesite=lax`;
    }
    router.push('/chat');
  };

  return (
    <main className="container flex min-h-[60vh] items-center justify-center py-12">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-lg border bg-card p-6 text-card-foreground shadow-sm">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold">Sign in</h1>
          <p className="text-sm text-muted-foreground">
            Use your raghub credentials and the workspace passphrase.
          </p>
        </div>
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="passphrase">Workspace passphrase</Label>
          <Input
            id="passphrase"
            type="password"
            autoComplete="off"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            required
          />
          <p className="text-xs text-muted-foreground">
            Required to unlock the encrypted workspace on disk.
          </p>
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Button type="submit" disabled={submitting} className="w-full">
          {submitting ? 'Signing in…' : 'Sign in'}
        </Button>
        <p className="text-xs text-muted-foreground">
          New here?{' '}
          <a className="underline" href="/onboarding">
            Create a workspace
          </a>
          .
        </p>
      </form>
    </main>
  );
}