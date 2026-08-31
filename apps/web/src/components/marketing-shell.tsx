"use client";

import * as React from "react";
import Link from "next/link";

import { Wordmark } from "@/components/wordmark";
import { Button } from "@/components/ui/button";
import { SunDim, Moon } from "@/lib/icons";
import { useTheme } from "next-themes";

interface MarketingShellProps {
  readonly children: React.ReactNode;
  readonly hideHeader?: boolean;
}

export function MarketingShell({ children, hideHeader }: MarketingShellProps) {
  return (
    <div className="relative isolate min-h-svh overflow-hidden">
      <div
        aria-hidden
        className="absolute inset-0 -z-10 gradient-mesh dark:gradient-mesh-dark"
      />
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 -z-10 h-[60vh] bg-gradient-to-b from-background/40 to-transparent"
      />
      {!hideHeader && <MarketingHeader />}
      <main className="flex flex-1 flex-col">{children}</main>
      <MarketingFooter />
    </div>
  );
}

function MarketingHeader() {
  const { setTheme, resolvedTheme } = useTheme();
  return (
    <header className="sticky top-0 z-20 border-b border-border/40 bg-background/60 backdrop-blur-md">
      <div className="container flex h-16 items-center justify-between">
        <Link href="/" className="inline-flex items-center">
          <Wordmark size="md" />
        </Link>
        <nav className="flex items-center gap-2">
          <Link
            href="/privacy"
            className="hidden text-sm text-muted-foreground transition-colors hover:text-foreground md:inline-block"
          >
            Privacy
          </Link>
          <Link
            href="/terms"
            className="hidden text-sm text-muted-foreground transition-colors hover:text-foreground md:inline-block"
          >
            Terms
          </Link>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
            aria-label="Toggle theme"
          >
            <SunDim className="size-4 dark:hidden" />
            <Moon className="hidden size-4 dark:block" />
          </Button>
          <Button asChild variant="ghost" size="sm">
            <Link href="/sign-in">Sign in</Link>
          </Button>
          <Button asChild size="sm">
            <Link href="/onboarding">Create workspace</Link>
          </Button>
        </nav>
      </div>
    </header>
  );
}

function MarketingFooter() {
  return (
    <footer className="border-t border-border/40 bg-background/40 py-10">
      <div className="container flex flex-col items-start justify-between gap-4 text-sm md:flex-row md:items-center">
        <div className="flex items-center gap-3 text-muted-foreground">
          <Wordmark size="sm" />
          <span>·</span>
          <span>Hybrid retrieval for teams.</span>
        </div>
        <nav className="flex flex-wrap items-center gap-x-6 gap-y-2 text-muted-foreground">
          <Link href="/privacy" className="hover:text-foreground transition-colors">
            Privacy
          </Link>
          <Link href="/terms" className="hover:text-foreground transition-colors">
            Terms
          </Link>
          <Link href="/sign-in" className="hover:text-foreground transition-colors">
            Sign in
          </Link>
          <Link href="/onboarding" className="hover:text-foreground transition-colors">
            Create workspace
          </Link>
        </nav>
      </div>
    </footer>
  );
}