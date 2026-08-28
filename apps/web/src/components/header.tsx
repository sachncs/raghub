'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

import { ThemeToggle } from '@/components/theme-toggle';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

interface NavLink {
  readonly href: string;
  readonly label: string;
}

const NAV: readonly NavLink[] = [
  { href: '/chat', label: 'Chat' },
  { href: '/documents', label: 'Documents' },
  { href: '/members', label: 'Members' },
  { href: '/settings', label: 'Settings' },
];

export interface HeaderProps {
  readonly children?: ReactNode;
}

export function Header(_props: HeaderProps) {
  const pathname = usePathname() ?? '';
  return (
    <header className="flex items-center justify-between border-b px-6 py-3">
      <Link href="/" className="text-sm font-semibold">
        raghub
      </Link>
      <nav aria-label="Primary" className="flex items-center gap-1 text-sm">
        {NAV.map((link) => {
          const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
          return (
            <Button
              key={link.href}
              asChild
              variant={active ? 'secondary' : 'ghost'}
              size="sm"
            >
              <Link href={link.href}>{link.label}</Link>
            </Button>
          );
        })}
        <Separator orientation="vertical" className="mx-1 h-5" />
        <Button
          variant="ghost"
          size="sm"
          onClick={() => void (async () => {
            await fetch('/api/proxy', {
              method: 'POST',
              headers: { 'x-raghub-path': '/v1/auth/logout' },
            });
            window.location.href = '/sign-in';
          })()}
        >
          Sign out
        </Button>
        <ThemeToggle />
      </nav>
    </header>
  );
}