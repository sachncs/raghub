import * as React from "react";
import Link from "next/link";

import { ArrowRight, ArrowUpRight, Sparkle } from "@/lib/icons";
import { Button } from "@/components/ui/button";
import { LiveQuery } from "@/components/marketing/live-query";

import { cn } from "@/lib/utils";

interface HeroProps {
  readonly className?: string;
}

export function Hero({ className }: HeroProps) {
  return (
    <section
      className={cn(
        "relative isolate overflow-hidden pb-20 pt-24 md:pb-28 md:pt-32",
        className
      )}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 grid-faint opacity-50 [mask-image:radial-gradient(ellipse_at_top,black,transparent_75%)]"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[60vh] bg-gradient-to-b from-indigo-500/10 via-background to-transparent"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -left-32 top-1/4 -z-10 size-[480px] rounded-full bg-indigo-500/15 blur-3xl"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -right-32 -top-12 -z-10 size-[420px] rounded-full bg-amber-500/10 blur-3xl"
      />

      <div className="container flex flex-col items-center px-6 text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/60 px-3 py-1 text-xs font-medium text-muted-foreground shadow-sm shadow-black/5 backdrop-blur-sm">
          <Sparkle className="size-3.5 text-primary" />
          <span>v1.1 · Revex Platform</span>
        </div>

        <h1 className="max-w-4xl text-balance text-display-1 text-foreground">
          Governed hybrid retrieval,
          <br className="hidden md:block" />{" "}
          <span className="text-gradient-brand">built for teams</span>{" "}
          that take access seriously.
        </h1>

        <p className="mt-6 max-w-2xl text-pretty text-lg text-muted-foreground md:text-xl">
          Vector, keyword, graph, memory, and web — fused behind one
          policy-aware engine. Deploy locally, govern centrally, audit
          everything.
        </p>

        <div className="mt-10 w-full max-w-3xl">
          <LiveQuery />
        </div>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
          <Button asChild size="lg" className="rounded-xl px-6">
            <Link href="/onboarding">
              Create workspace
              <ArrowRight className="size-4" />
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="rounded-xl px-6">
            <Link href="/docs/self-host">
              Self-host on your infra
              <ArrowUpRight className="size-4" />
            </Link>
          </Button>
        </div>

        <p className="mt-6 inline-flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
            SOC 2 Type II
          </span>
          <span aria-hidden>·</span>
          <span>ISO 27001</span>
          <span aria-hidden>·</span>
          <span>HIPAA-ready</span>
          <span aria-hidden>·</span>
          <span>MIT-licensed core</span>
        </p>
      </div>
    </section>
  );
}