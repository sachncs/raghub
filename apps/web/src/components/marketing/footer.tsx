import * as React from "react";
import Link from "next/link";

import { LogoMark } from "@/components/marketing/logo-mark";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Moon, SunDim } from "@/lib/icons";
import { useTheme } from "next-themes";

import { cn } from "@/lib/utils";

interface FooterColumn {
  readonly title: string;
  readonly links: ReadonlyArray<{ label: string; href: string }>;
}

const COLUMNS: readonly FooterColumn[] = [
  {
    title: "Product",
    links: [
      { label: "Hybrid retrieval", href: "/#capabilities" },
      { label: "Reranking", href: "/#capabilities" },
      { label: "Generation", href: "/#capabilities" },
      { label: "Governance", href: "/#policy" },
      { label: "Pricing", href: "/pricing" },
    ],
  },
  {
    title: "Solutions",
    links: [
      { label: "Knowledge base", href: "/#capabilities" },
      { label: "Internal copilots", href: "/#capabilities" },
      { label: "Customer support", href: "/#capabilities" },
      { label: "Compliance & legal", href: "/#policy" },
      { label: "Operational intel", href: "/#platform" },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "Documentation", href: "/docs" },
      { label: "Self-host guide", href: "/docs/self-host" },
      { label: "API reference", href: "/docs/api" },
      { label: "Changelog", href: "/changelog" },
      { label: "Status", href: "/status" },
    ],
  },
  {
    title: "Company",
    links: [
      { label: "About", href: "/about" },
      { label: "Security", href: "/#policy" },
      { label: "Privacy", href: "/privacy" },
      { label: "Terms", href: "/terms" },
      { label: "Contact", href: "mailto:hello@revex.io" },
    ],
  },
];

interface FooterProps {
  readonly className?: string;
}

export function Footer({ className }: FooterProps) {
  return (
    <footer
      className={cn(
        "relative isolate border-t border-border/40 bg-background/40 backdrop-blur-sm",
        className
      )}
    >
      <div className="container px-6 py-16">
        <div className="grid gap-12 md:grid-cols-[2fr_3fr] md:gap-16">
          <div className="flex flex-col gap-6">
            <Link
              href="/"
              aria-label="Revex home"
              className="inline-flex items-center gap-2.5"
            >
              <LogoMark size="md" />
              <span className="text-lg font-semibold italic tracking-tight">
                re
                <span className="not-italic text-primary">›</span>
                ex
              </span>
            </Link>
            <p className="max-w-sm text-sm leading-relaxed text-muted-foreground">
              Governed hybrid retrieval for teams that take access seriously.
              Self-host or cloud. MIT-licensed core.
            </p>
            <Badge variant="outline" className="w-fit gap-2">
              <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
              All systems operational
            </Badge>
            <ThemeRow />
          </div>

          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            {COLUMNS.map((col, i) => (
              <React.Fragment key={col.title}>
                {i > 0 && <Separator orientation="vertical" className="hidden md:block" />}
                <div>
                  <h3 className="text-eyebrow text-muted-foreground">
                    {col.title}
                  </h3>
                  <ul className="mt-4 flex flex-col gap-2.5">
                    {col.links.map((link) => (
                      <li key={link.href + link.label}>
                        <Link
                          href={link.href}
                          className="text-sm text-foreground/80 transition-colors hover:text-foreground"
                        >
                          {link.label}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              </React.Fragment>
            ))}
          </div>
        </div>

        <Separator className="my-8" />

        <div className="flex flex-col items-start justify-between gap-4 text-xs text-muted-foreground md:flex-row md:items-center">
          <span>
            © {new Date().getFullYear()} Revex · Governed retrieval for serious teams.
          </span>
          <span className="inline-flex items-center gap-3 font-mono">
            <span className="inline-flex items-center gap-1.5">
              <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
              all systems operational
            </span>
            <span aria-hidden>·</span>
            <span>v1.1.0</span>
          </span>
        </div>
      </div>
    </footer>
  );
}

function ThemeRow() {
  const { setTheme, resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <button
        type="button"
        onClick={() => setTheme("light")}
        aria-label="Switch to light theme"
        aria-pressed={!isDark}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-background/40 px-2.5 py-1 transition-colors hover:border-border hover:text-foreground",
          !isDark && "border-primary/60 text-foreground"
        )}
      >
        <SunDim className="size-3.5" />
        Light
      </button>
      <button
        type="button"
        onClick={() => setTheme("dark")}
        aria-label="Switch to dark theme"
        aria-pressed={isDark}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-background/40 px-2.5 py-1 transition-colors hover:border-border hover:text-foreground",
          isDark && "border-primary/60 text-foreground"
        )}
      >
        <Moon className="size-3.5" />
        Dark
      </button>
    </div>
  );
}