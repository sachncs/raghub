import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { MarketingShell } from '@/components/marketing-shell';

export default function MarketingNotFound() {
  return (
    <MarketingShell>
      <div className="container flex min-h-[60vh] items-center justify-center py-20">
        <div className="max-w-md text-center">
          <p className="mb-1 text-sm font-medium text-muted-foreground">404</p>
          <h1 className="mb-2 text-3xl font-semibold">Page not found</h1>
          <p className="mb-6 text-sm text-muted-foreground">
            We couldn&apos;t find that page. Try heading back home.
          </p>
          <Button asChild>
            <Link href="/">Back to home</Link>
          </Button>
        </div>
      </div>
    </MarketingShell>
  );
}