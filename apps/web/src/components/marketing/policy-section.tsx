import * as React from "react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

import { cn } from "@/lib/utils";

interface PolicySectionProps {
  readonly className?: string;
}

interface PolicyRow {
  readonly id: string;
  readonly title: string;
  readonly source: string;
  readonly decision: "allow" | "redact" | "deny";
  readonly reason: string;
}

const POLICY_ROWS: readonly PolicyRow[] = [
  {
    id: "p1",
    title: "Q3-contract.pdf",
    source: "SharePoint · Legal",
    decision: "allow",
    reason: "role:legal",
  },
  {
    id: "p2",
    title: "Pricing memo · 2026-Q1",
    source: "Notion · Finance",
    decision: "redact",
    reason: "redact:[price]",
  },
  {
    id: "p3",
    title: "Vendor risk register",
    source: "Drive · Compliance",
    decision: "allow",
    reason: "role:compliance",
  },
  {
    id: "p4",
    title: "External NDA · Acme",
    source: "Drive · Legal",
    decision: "deny",
    reason: "group:external",
  },
];

const POLICY_LINES: ReadonlyArray<{ tokens: ReadonlyArray<{ text: string; cls: string }> }> = [
  { tokens: [{ text: 'policy', cls: "text-indigo-400" }, { text: ": ", cls: "text-foreground/70" }, { text: '"Q3-contract.pdf"', cls: "text-amber-500" }] },
  { tokens: [{ text: "  ", cls: "text-foreground/70" }, { text: "allow", cls: "text-indigo-400" }, { text: ": ", cls: "text-foreground/70" }, { text: '[ "role:legal", "user:ceo" ]', cls: "text-amber-500" }] },
  { tokens: [{ text: "  ", cls: "text-foreground/70" }, { text: "deny", cls: "text-indigo-400" }, { text: ":  ", cls: "text-foreground/70" }, { text: '[ "group:external" ]', cls: "text-amber-500" }] },
  { tokens: [{ text: "  ", cls: "text-foreground/70" }, { text: "redact", cls: "text-indigo-400" }, { text: ": ", cls: "text-foreground/70" }, { text: '[ "price", "margin" ]', cls: "text-amber-500" }] },
  { tokens: [{ text: "  ", cls: "text-foreground/70" }, { text: "require_audit", cls: "text-indigo-400" }, { text: ": ", cls: "text-foreground/70" }, { text: "true", cls: "text-foreground/80" }] },
  { tokens: [{ text: "---", cls: "text-muted-foreground" }] },
];

const DECISION_STYLES: Record<PolicyRow["decision"], string> = {
  allow: "bg-emerald-500/15 text-emerald-500",
  redact: "bg-amber-500/15 text-amber-500",
  deny: "bg-rose-500/15 text-rose-500",
};

const DECISION_LABEL: Record<PolicyRow["decision"], string> = {
  allow: "allow",
  redact: "redact",
  deny: "deny",
};

export function PolicySection({ className }: PolicySectionProps) {
  return (
    <section
      id="policy"
      className={cn("relative isolate overflow-hidden py-16 md:py-28", className)}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 grid-faint opacity-40 [mask-image:radial-gradient(ellipse_at_center,black,transparent_75%)]"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -left-32 top-1/3 -z-10 size-[420px] rounded-full bg-emerald-500/10 blur-3xl"
      />
      <div className="container px-6">
        <div className="mx-auto max-w-3xl text-center">
          <Badge variant="outline" className="mb-3 gap-2">
            <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
            Governance
          </Badge>
          <h2 className="text-h1 text-balance text-foreground">
            Document-level access control, enforced before ranking.
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-pretty text-base text-muted-foreground md:text-lg">
            Policies are first-class. They govern which sources are queried,
            which documents can return, and which fields survive to the answer
            — per requester, per query.
          </p>
        </div>

        <div className="mt-12 grid gap-6 lg:grid-cols-[5fr_4fr]">
          <PolicyEditor />
          <PolicyScope />
        </div>
      </div>
    </section>
  );
}

function PolicyEditor() {
  return (
    <Card className="overflow-hidden border-border/60 bg-card/50 shadow-2xl shadow-black/10 backdrop-blur-md">
      <CardContent className="p-0">
        <div className="flex items-center justify-between border-b border-border/40 bg-background/40 px-4 py-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="size-2 rounded-full bg-emerald-500" aria-hidden />
            policy.revex · main
          </div>
          <span className="text-eyebrow text-muted-foreground">Policy editor</span>
        </div>
        <pre className="overflow-x-auto p-5 font-mono text-sm leading-relaxed text-foreground/90">
          <code>
            {POLICY_LINES.map((line, i) => (
              <span key={i} className="block">
                {line.tokens.map((tok, j) => (
                  <span key={j} className={tok.cls}>
                    {tok.text}
                  </span>
                ))}
              </span>
            ))}
            <span className="mt-2 inline-block size-2 animate-pulse rounded-full bg-emerald-500 align-middle" aria-hidden />
          </code>
        </pre>
        <div className="flex items-center justify-between gap-3 border-t border-border/40 bg-background/40 px-4 py-2 text-xs text-muted-foreground">
          <Badge variant="outline" className="gap-1.5">
            <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
            Saved · policy.lint passed
          </Badge>
          <span className="font-mono">+ rule</span>
        </div>
      </CardContent>
    </Card>
  );
}

function PolicyScope() {
  return (
    <div className="flex flex-col gap-3">
      <Card>
        <CardContent className="flex items-center justify-between gap-3 p-4">
          <div className="flex items-center gap-3">
            <span className="inline-flex size-9 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-500">
              <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-500">
                OK
              </Badge>
            </span>
            <div>
              <div className="text-sm font-medium text-foreground">Policy decision</div>
              <div className="text-xs text-muted-foreground">
                3 allow · 1 redact · 1 deny
              </div>
            </div>
          </div>
          <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-500">
            in scope
          </Badge>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-0">
          <ul className="flex flex-col divide-y divide-border/40">
            {POLICY_ROWS.map((row) => (
              <li
                key={row.id}
                className="flex items-center gap-3 p-3"
              >
                <span
                  className={cn(
                    "inline-flex h-7 items-center justify-center rounded-md px-2 text-[11px] font-medium uppercase tracking-wider",
                    DECISION_STYLES[row.decision]
                  )}
                >
                  {DECISION_LABEL[row.decision]}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-foreground">
                    {row.title}
                  </div>
                  <div className="truncate text-xs text-muted-foreground">
                    {row.source}
                  </div>
                </div>
                <span className="hidden rounded-md border border-border/40 bg-background/50 px-2 py-0.5 font-mono text-[11px] text-muted-foreground sm:inline">
                  {row.reason}
                </span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="default-deny" className="rounded-2xl border border-dashed border-border/60 bg-card/20 px-4">
          <AccordionTrigger className="text-sm font-medium">
            Default-deny safe mode
          </AccordionTrigger>
          <AccordionContent className="text-xs leading-relaxed text-muted-foreground">
            Documents without a matching rule are excluded from retrieval
            results, never silently allowed. Override per-document with an
            explicit <code className="rounded bg-muted px-1 font-mono">allow</code> rule.
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}