import Link from 'next/link';

export default function Home() {
  return (
    <main className="container flex min-h-screen flex-col items-center justify-center py-16 text-center">
      <h1 className="mb-3 text-4xl font-bold tracking-tight">raghub</h1>
      <p className="mb-6 max-w-md text-muted-foreground">
        Multi-user RAG on Strands Agents. One orchestrator, three patterns, per-user strategy.
      </p>
      <nav className="flex gap-3">
        <Link href="/sign-in" className="rounded-md bg-primary px-4 py-2 text-primary-foreground">Sign in</Link>
        <Link href="/onboarding" className="rounded-md border px-4 py-2">Onboard</Link>
      </nav>
    </main>
  );
}