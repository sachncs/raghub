'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldDescription, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';

export default function SignInPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passphrase, setPassphrase] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const submit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    setSubmitting(true);
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
      toast.error(body?.error?.message ?? 'login failed');
      return;
    }
    document.cookie = `raghub_token=${body.token}; path=/; max-age=86400; samesite=lax`;
    if (passphrase.length > 0) {
      document.cookie = `raghub_passphrase=${encodeURIComponent(passphrase)}; path=/; max-age=86400; samesite=lax`;
    }
    toast.success('Signed in');
    router.push('/chat');
  };

  return (
    <div className="flex min-h-[60vh] items-center justify-center py-12">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Sign in</CardTitle>
          <CardDescription>
            Use your raghub credentials and the workspace passphrase.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-4">
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="email">Email</FieldLabel>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="password">Password</FieldLabel>
                <Input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="passphrase">Workspace passphrase</FieldLabel>
                <Input
                  id="passphrase"
                  type="password"
                  autoComplete="off"
                  value={passphrase}
                  onChange={(e) => setPassphrase(e.target.value)}
                  required
                />
                <FieldDescription>
                  Required to unlock the encrypted workspace on disk.
                </FieldDescription>
              </Field>
            </FieldGroup>
            <Button type="submit" disabled={submitting} className="w-full">
              {submitting ? 'Signing in…' : 'Sign in'}
            </Button>
            <p className="text-center text-xs text-muted-foreground">
              New here?{' '}
              <Link className="underline" href="/onboarding">
                Create a workspace
              </Link>
              .
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}