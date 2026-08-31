import Link from 'next/link';
import { Button } from '@/components/ui/button';

export default function AppNotFound() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center px-6">
      <div className="max-w-md text-center">
        <p className="mb-1 text-sm font-medium text-muted-foreground">404</p>
        <h1 className="mb-2 text-3xl font-semibold">Page not found</h1>
        <p className="mb-6 text-sm text-muted-foreground">
          The page you were looking for doesn&apos;t exist in this workspace.
        </p>
        <Button asChild>
          <Link href="/chat">Back to chat</Link>
        </Button>
      </div>
    </div>
  );
}