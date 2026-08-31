"use client";

import * as React from "react";
import Link from "next/link";

import { ArrowLeft } from "@/lib/icons";
import { LogoMark } from "@/components/marketing/logo-mark";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

import { cn } from "@/lib/utils";

interface AuthCardProps {
  readonly className?: string;
  readonly left?: React.ReactNode;
  readonly children: React.ReactNode;
  readonly trustPills?: readonly string[];
  readonly showBrand?: boolean;
}

export function AuthCard({
  className,
  left,
  children,
  trustPills,
  showBrand = true,
}: AuthCardProps) {
  const hasTrust = Array.isArray(trustPills) && trustPills.length > 0;

  return (
    <div
      className={cn(
        "relative isolate flex min-h-svh w-full flex-col overflow-hidden",
        className
      )}
    >
      <div
        aria-hidden
        className="absolute inset-0 -z-10 grid-faint opacity-60"
      />
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 -z-10 h-[60vh] bg-gradient-to-b from-background/40 to-transparent"
      />
      <div
        aria-hidden
        className="absolute -left-32 top-1/4 -z-10 hidden size-[480px] rounded-full bg-indigo-500/10 blur-3xl md:block"
      />
      <div
        aria-hidden
        className="absolute -right-32 bottom-0 -z-10 hidden size-[520px] rounded-full bg-amber-500/10 blur-3xl md:block"
      />

      <div className="container flex flex-1 items-stretch px-6 py-10 md:py-16">
        <div className="grid w-full gap-10 md:grid-cols-[5fr_4fr] md:gap-16">
          {left !== undefined && (
            <aside className="hidden flex-col justify-between rounded-2xl border border-border/40 bg-card/30 p-10 backdrop-blur-sm md:flex">
              {left}
            </aside>
          )}
          <main className="flex items-center">
            <Card className="w-full max-w-md border-border/60 bg-card/60 shadow-2xl shadow-black/10 backdrop-blur-md">
              <CardContent className="space-y-6 p-6 md:p-8">
                {showBrand && (
                  <div className="flex items-center justify-between">
                    <Link
                      href="/"
                      aria-label="Revex home"
                      className="inline-flex items-center gap-2.5"
                    >
                      <LogoMark size="sm" />
                      <span className="text-base font-semibold italic tracking-tight">
                        re
                        <span className="not-italic text-primary">›</span>
                        ex
                      </span>
                    </Link>
                    <Link
                      href="/"
                      className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <ArrowLeft className="size-3.5" />
                      Back home
                    </Link>
                  </div>
                )}

                {children}

                {hasTrust && (
                  <>
                    <Separator />
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {trustPills?.map((pill) => (
                        <span
                          key={pill}
                          className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/50 px-2.5 py-1 text-muted-foreground"
                        >
                          <span
                            className="size-1.5 rounded-full bg-emerald-500"
                            aria-hidden
                          />
                          {pill}
                        </span>
                      ))}
                    </div>
                  </>
                )}

                {!showBrand && (
                  <div className="text-center text-xs text-muted-foreground">
                    <Link
                      href="/"
                      className="inline-flex items-center gap-1 transition-colors hover:text-foreground"
                    >
                      <ArrowLeft className="size-3.5" />
                      Back to Revex
                    </Link>
                  </div>
                )}
              </CardContent>
            </Card>
          </main>
        </div>
      </div>
    </div>
  );
}

interface AuthTrustPillsProps {
  readonly pills: readonly string[];
  readonly className?: string;
}

export function AuthTrustPills({ pills, className }: AuthTrustPillsProps) {
  return (
    <ul className={cn("flex flex-wrap items-center gap-2 text-xs", className)}>
      {pills.map((pill) => (
        <li
          key={pill}
          className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/50 px-2.5 py-1 text-muted-foreground"
        >
          <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
          {pill}
        </li>
      ))}
    </ul>
  );
}

export { Button as AuthButton };