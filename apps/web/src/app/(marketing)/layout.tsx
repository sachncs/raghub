import * as React from 'react';
import { ThemeProvider } from '@/components/theme-provider';
import { MarketingShell } from '@/components/marketing-shell';

export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ThemeProvider defaultTheme="light">
      <MarketingShell>{children}</MarketingShell>
    </ThemeProvider>
  );
}