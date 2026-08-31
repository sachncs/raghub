'use client';

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { MarketingShell } from '@/components/marketing-shell';

export default function MarketingError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <MarketingShell>
      <div className="container flex min-h-[60vh] items-center justify-center py-20">
        <div className="max-w-md text-center">
          <h1 className="mb-2 text-2xl font-semibold">Something went wrong</h1>
          <p className="mb-6 text-sm text-muted-foreground">
            {error.message || 'An unexpected error occurred.'}
          </p>
          <Button onClick={reset}>Try again</Button>
        </div>
      </div>
    </MarketingShell>
  );
}