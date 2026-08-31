import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

import { cn } from "@/lib/utils";

interface TrustStripProps {
  readonly className?: string;
}

interface KpiTile {
  readonly value: string;
  readonly label: string;
  readonly detail: string;
}

const KPIS: readonly KpiTile[] = [
  { value: "12 ms", label: "p95 retrieval latency", detail: "across 9 sources" },
  { value: "9", label: "retrieval sources fused", detail: "vector · bm25 · graph · memory · web · rerank" },
  { value: "100%", label: "queries policy-enforced", detail: "document-level ACLs, every call" },
  { value: "0 B", label: "data leaves your host", detail: "self-hosted or air-gapped" },
];

interface GovernanceClaim {
  readonly title: string;
  readonly detail: string;
}

const GOVERNANCE: readonly GovernanceClaim[] = [
  {
    title: "Document-level ACLs",
    detail: "Per-user, per-group, per-tenant policies. Enforced before ranking.",
  },
  {
    title: "Per-user strategy routing",
    detail: "Tune retrieval per role without changing code.",
  },
  {
    title: "Sealed SQLite + passphrase",
    detail: "Every workspace is one encrypted file. Yours to keep.",
  },
];

export function TrustStrip({ className }: TrustStripProps) {
  return (
    <section
      id="trust"
      className={cn("relative isolate py-16 md:py-24", className)}
    >
      <div className="container px-6">
        <div className="mx-auto max-w-3xl text-center">
          <Badge variant="outline" className="mb-3 gap-2">
            <span className="size-1.5 rounded-full bg-muted-foreground" aria-hidden />
            Operate with proof
          </Badge>
          <h2 className="text-h1 text-balance text-foreground">
            Built for teams who treat retrieval like infrastructure.
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-pretty text-base text-muted-foreground md:text-lg">
            Revex runs where you run. Every query is policy-gated, every answer
            is auditable, every byte stays on your perimeter.
          </p>
        </div>

        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {KPIS.map((kpi) => (
            <Card key={kpi.label} className="transition-all hover:-translate-y-0.5 hover:bg-card/70">
              <CardContent className="space-y-1 p-6">
                <div className="text-display-2 font-semibold tracking-tight text-foreground">
                  {kpi.value}
                </div>
                <div className="text-sm font-medium text-foreground/80">
                  {kpi.label}
                </div>
                <div className="text-xs text-muted-foreground">{kpi.detail}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-8 grid gap-3 md:grid-cols-3">
          {GOVERNANCE.map((claim) => (
            <Card key={claim.title} className="bg-card/30">
              <CardContent className="flex items-start gap-3 p-4">
                <Badge variant="secondary" className="mt-0.5 bg-emerald-500/10 text-emerald-500">
                  Governance
                </Badge>
                <div className="flex flex-col gap-0.5">
                  <span className="text-sm font-medium text-foreground">
                    {claim.title}
                  </span>
                  <span className="text-xs leading-relaxed text-muted-foreground">
                    {claim.detail}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <Badge variant="outline" className="rounded-sm font-normal">
              OpenAI
            </Badge>
            <Badge variant="outline" className="rounded-sm font-normal">
              Anthropic
            </Badge>
            <Badge variant="outline" className="rounded-sm font-normal">
              Cohere
            </Badge>
            <Badge variant="outline" className="rounded-sm font-normal">
              Voyage
            </Badge>
            <Badge variant="outline" className="rounded-sm font-normal">
              Bedrock
            </Badge>
            <Badge variant="outline" className="rounded-sm font-normal">
              LiteLLM
            </Badge>
          </span>
        </div>
      </div>
    </section>
  );
}