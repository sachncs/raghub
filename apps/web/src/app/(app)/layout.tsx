import * as React from 'react';
import { cookies } from 'next/headers';
import { ThemeProvider } from '@/components/theme-provider';
import { AppShell } from '@/components/app-shell';

async function fetchMe(): Promise<{ email?: string | null; displayName?: string | null } | null> {
  const token = (await cookies()).get('revex_session')?.value;
  if (!token) return null;
  try {
    const res = await fetch(
      `${process.env['REVEX_API_BASE'] ?? 'http://localhost:3000'}/v1/me`,
      {
        headers: { authorization: `Bearer ${token}` },
        cache: 'no-store',
      }
    );
    if (!res.ok) return null;
    const body = (await res.json()) as { user?: { email?: string; displayName?: string } };
    return body.user ?? null;
  } catch {
    return null;
  }
}

export default async function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await fetchMe();
  return (
    <ThemeProvider defaultTheme="dark">
      <AppShell user={user ?? undefined}>{children}</AppShell>
    </ThemeProvider>
  );
}