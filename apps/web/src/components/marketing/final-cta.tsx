import * as React from "react";
import Link from "next/link";

import { ArrowRight, ArrowUpRight, ChatCircle } from "@/lib/icons";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { LogoMark } from "@/components/marketing/logo-mark";

import { cn } from "@/lib/utils";

interface FinalCtaProps {
  readonly className?: string;
}

export function FinalCta({ className }: FinalCtaProps) {
  return (
    <section
      id="cta"
      className={cn(
        "relative isolate overflow-hidden py-20 md:py-32",
        className
      )}
    >
      <div className="container px-6">
        <Card className="relative overflow-hidden border-border/60 bg-card/40 shadow-2xl shadow-black/10 backdrop-blur-md">
          <CardContent className="flex flex-col items-center px-8 py-14 text-center md:px-16 md:py-20">
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 -z-10 gradient-conic opacity-30 blur-3xl"
            />
            <div
              aria-hidden
              className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent"
            />
            <div
              aria-hidden
              className="pointer-events-none absolute -right-32 -top-32 -z-10 size-[420px] rounded-full bg-amber-500/15 blur-3xl"
            />
            <div
              aria-hidden
              className="pointer-events-none absolute -bottom-32 -left-32 -z-10 size-[420px] rounded-full bg-indigo-500/15 blur-3xl"
            />

            <LogoMark size="lg" />
            <h2 className="mt-6 max-w-3xl text-balance text-display-2 text-foreground">
              Retrieval you can run,
              <br className="hidden md:block" />{" "}
              <span className="text-gradient-brand">audit, and trust.</span>
            </h2>
            <p className="mt-4 max-w-xl text-pretty text-base text-muted-foreground md:text-lg">
              Three ways to start. Same engine. Same policy layer. Same audit trail.
            </p>

            <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
              <Button asChild size="lg" className="rounded-xl px-6">
                <Link href="/onboarding">
                  Create workspace
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="rounded-xl px-6">
                <Link href="/docs/self-host">
                  Self-host guide
                  <ArrowUpRight className="size-4" />
                </Link>
              </Button>
              <Button asChild variant="ghost" size="lg" className="rounded-xl px-6">
                <Link href="mailto:hello@revex.io">
                  <ChatCircle className="size-4" />
                  Talk to platform team
                </Link>
              </Button>
            </div>

            <p className="mt-8 inline-flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
              <span>Free, MIT-licensed core.</span>
              <span aria-hidden>·</span>
              <span>No vendor lock-in.</span>
              <span aria-hidden>·</span>
              <span>Bring your own LLM.</span>
            </p>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}