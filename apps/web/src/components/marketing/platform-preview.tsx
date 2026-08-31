import * as React from "react";

import {
  BadgeIcon,
  CircuitBoard,
  Clock,
  Database,
  Eye,
  FileText,
  Gauge,
  GitBranch,
  KeyRound,
  Layers,
  LineChart,
  Plug,
  Radar,
  ServerCog,
  ShieldCheck,
  Workflow,
} from "@/lib/icons";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { cn } from "@/lib/utils";

interface PlatformPreviewProps {
  readonly className?: string;
}

interface SourceRow {
  readonly id: string;
  readonly name: string;
  readonly kind: string;
  readonly status: "ok" | "warn" | "down";
}

const SOURCES: readonly SourceRow[] = [
  { id: "s1", name: "vector · bge-large", kind: "Dense", status: "ok" },
  { id: "s2", name: "bm25 · contracts", kind: "Lexical", status: "ok" },
  { id: "s3", name: "graph · entities", kind: "GraphRAG", status: "ok" },
  { id: "s4", name: "memory · sessions", kind: "Memory", status: "ok" },
  { id: "s5", name: "reranker · cohere-rerank-3", kind: "Rerank", status: "ok" },
];

const ROUTING: ReadonlyArray<{
  readonly id: string;
  readonly label: string;
  readonly weight: number;
}> = [
  { id: "r1", label: "vector", weight: 0.34 },
  { id: "r2", label: "bm25", weight: 0.21 },
  { id: "r3", label: "graph", weight: 0.18 },
  { id: "r4", label: "memory", weight: 0.15 },
  { id: "r5", label: "rerank", weight: 0.12 },
];

interface RetrievalRow {
  readonly id: string;
  readonly title: string;
  readonly source: string;
  readonly score: number;
  readonly status: "allow" | "redact" | "deny";
}

const RETRIEVAL: readonly RetrievalRow[] = [
  { id: "t1", title: "Q3-contract.pdf", source: "SharePoint · Legal", score: 0.94, status: "allow" },
  { id: "t2", title: "Pricing memo · 2026-Q1", source: "Notion · Finance", score: 0.88, status: "redact" },
  { id: "t3", title: "Vendor risk register", source: "Drive · Compliance", score: 0.81, status: "allow" },
  { id: "t4", title: "External NDA · Acme", source: "Drive · Legal", score: 0.74, status: "deny" },
];

const AUDIT: readonly { id: string; time: string; actor: string; action: string }[] = [
  { id: "a1", time: "12:04:18", actor: "user:jane@revex.io", action: "query · policy=legal · 12 hits" },
  { id: "a2", time: "12:04:18", actor: "system", action: "acl gate · 12/12 allow · 0 redact" },
  { id: "a3", time: "12:04:17", actor: "reranker:cohere", action: "rerank · 12 → 5 · 9ms" },
  { id: "a4", time: "12:04:16", actor: "graph:entities", action: "expand · 4 hops · 8 nodes" },
];

interface PlatformTile {
  readonly title: string;
  readonly description: string;
  readonly icon: React.ComponentType<{ className?: string }>;
  readonly status: "stable" | "beta" | "alpha";
}

const PLATFORM: readonly PlatformTile[] = [
  {
    title: "Workspaces",
    description: "Sealed SQLite per workspace. Open with a passphrase.",
    icon: KeyRound,
    status: "stable",
  },
  {
    title: "Indexing",
    description: "Connectors, schedulers, deduplication, change detection.",
    icon: Plug,
    status: "stable",
  },
  {
    title: "Sources",
    description: "Files, S3, Notion, Confluence, Slack, databases, webhooks.",
    icon: Database,
    status: "stable",
  },
  {
    title: "Memory",
    description: "Session context with auto-summarisation, TTL, receipts.",
    icon: GitBranch,
    status: "stable",
  },
  {
    title: "Query routing",
    description: "Per-user strategy. Fan-out to 5 sources, RRF fusion.",
    icon: Workflow,
    status: "stable",
  },
  {
    title: "Observability",
    description: "Latency, hit-rate, retrieval cost, answer faithfulness.",
    icon: LineChart,
    status: "beta",
  },
  {
    title: "Audit trail",
    description: "Append-only ledger of every retrieval and policy decision.",
    icon: ShieldCheck,
    status: "stable",
  },
];

const STATUS_STYLES: Record<PlatformTile["status"], string> = {
  stable: "bg-emerald-500/15 text-emerald-500",
  beta: "bg-amber-500/15 text-amber-500",
  alpha: "bg-rose-500/15 text-rose-500",
};

export function PlatformPreview({ className }: PlatformPreviewProps) {
  return (
    <section
      id="platform"
      className={cn("relative isolate py-16 md:py-24", className)}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 grid-faint opacity-30 [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]"
      />
      <div className="container px-6">
        <div className="mx-auto max-w-3xl text-center">
          <Badge variant="outline" className="mb-3 gap-2">
            <span className="size-1.5 rounded-full bg-indigo-500" aria-hidden />
            Platform
          </Badge>
          <h2 className="text-h1 text-balance text-foreground">
            The retrieval layer your team will actually own.
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-pretty text-base text-muted-foreground md:text-lg">
            Eight subsystems, one engine. Designed to be inspected, extended,
            and operated by humans — not black boxes.
          </p>
        </div>

        <div className="mt-12">
          <ProductMock />
        </div>

        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {PLATFORM.map((tile) => (
            <Card
              key={tile.title}
              className="group transition-all hover:-translate-y-0.5 hover:bg-card/80 hover:shadow-lg hover:shadow-black/5"
            >
              <CardContent className="flex flex-col gap-3 p-5">
                <div className="flex items-center justify-between">
                  <span className="inline-flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <tile.icon className="size-4" />
                  </span>
                  <Badge variant="outline" className={cn("uppercase tracking-wider", STATUS_STYLES[tile.status])}>
                    {tile.status}
                  </Badge>
                </div>
                <h3 className="text-base font-semibold text-foreground">
                  {tile.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {tile.description}
                </p>
              </CardContent>
            </Card>
          ))}
          <Card className="border-dashed bg-card/20 transition-all hover:-translate-y-0.5 hover:bg-card/40">
            <CardContent className="flex flex-col gap-3 p-5">
              <div className="flex items-center justify-between">
                <span className="inline-flex size-9 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                  <ServerCog className="size-4" />
                </span>
                <Badge variant="outline" className="bg-muted/50 text-muted-foreground">
                  self-host
                </Badge>
              </div>
              <h3 className="text-base font-semibold text-foreground">
                Self-host
              </h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Single binary. Docker, Podman, systemd unit, or k8s. Air-gap-friendly.
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-8 grid gap-3 md:grid-cols-3">
          <Card className="bg-card/30">
            <CardContent className="flex items-center gap-3 p-4 text-xs">
              <Gauge className="size-4 text-indigo-500" />
              <span className="text-muted-foreground">
                <span className="text-foreground">Probes and metrics</span>{" "}
                emit OpenTelemetry. Wire into Datadog, Honeycomb, Grafana.
              </span>
            </CardContent>
          </Card>
          <Card className="bg-card/30">
            <CardContent className="flex items-center gap-3 p-4 text-xs">
              <Eye className="size-4 text-indigo-500" />
              <span className="text-muted-foreground">
                <span className="text-foreground">Trace every retrieval</span>{" "}
                with policy decision, sources, and latency breakdown.
              </span>
            </CardContent>
          </Card>
          <Card className="bg-card/30">
            <CardContent className="flex items-center gap-3 p-4 text-xs">
              <CircuitBoard className="size-4 text-indigo-500" />
              <span className="text-muted-foreground">
                <span className="text-foreground">Webhook hooks</span> fire on ingest, query, and audit events.
              </span>
            </CardContent>
          </Card>
        </div>

        <p className="mt-6 flex items-center gap-2 text-xs text-muted-foreground">
          <FileText className="size-3.5" />
          See the platform reference at{" "}
          <code className="rounded bg-muted/40 px-1.5 py-0.5 font-mono">/docs/platform</code>.
        </p>
      </div>
    </section>
  );
}

function ProductMock() {
  return (
    <Card className="overflow-hidden border-border/60 bg-card/40 shadow-2xl shadow-black/10 backdrop-blur-md">
      <CardContent className="p-0">
        <div className="flex items-center justify-between border-b border-border/40 bg-background/40 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <span className="size-2.5 rounded-full bg-rose-500/70" aria-hidden />
            <span className="size-2.5 rounded-full bg-amber-500/70" aria-hidden />
            <span className="size-2.5 rounded-full bg-emerald-500/70" aria-hidden />
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5 rounded-md bg-background/60 px-2 py-0.5 font-mono">
              <Eye className="size-3" />
              workspace · prod
            </span>
          </div>
          <Badge variant="outline" className="gap-1.5">
            <ShieldCheck className="size-3 text-emerald-500" />
            sealed
          </Badge>
        </div>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="h-auto w-full justify-start gap-1 rounded-none border-b border-border/40 bg-background/40 px-4 py-2">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="sources">Sources</TabsTrigger>
            <TabsTrigger value="routing">Routing</TabsTrigger>
            <TabsTrigger value="retrieval">Retrieval</TabsTrigger>
            <TabsTrigger value="audit">Audit</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="m-0 grid grid-cols-1 gap-px bg-border/40 md:grid-cols-12">
            <Pane title="Sources" icon={Layers} span={3}>
              <ul className="flex flex-col gap-1.5">
                {SOURCES.map((s) => (
                  <li
                    key={s.id}
                    className="flex items-center gap-2 rounded-md border border-border/40 bg-background/40 px-2 py-1.5 text-xs"
                  >
                    <span
                      className={cn(
                        "size-2 shrink-0 rounded-full",
                        s.status === "ok" && "bg-emerald-500",
                        s.status === "warn" && "bg-amber-500",
                        s.status === "down" && "bg-rose-500"
                      )}
                      aria-hidden
                    />
                    <span className="truncate text-foreground/80">{s.name}</span>
                    <span className="ml-auto text-muted-foreground">{s.kind}</span>
                  </li>
                ))}
              </ul>
            </Pane>
            <Pane title="Query routing" icon={BadgeIcon} span={3}>
              <ul className="flex flex-col gap-2">
                {ROUTING.map((r) => (
                  <li key={r.id} className="flex items-center gap-2 text-xs">
                    <span className="w-16 text-muted-foreground">{r.label}</span>
                    <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted/40">
                      <span
                        className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-indigo-500 to-amber-500"
                        style={{ width: `${Math.round(r.weight * 100)}%` }}
                      />
                    </span>
                    <span className="w-10 text-right font-mono text-foreground/70">
                      {(r.weight * 100).toFixed(0)}
                    </span>
                  </li>
                ))}
              </ul>
            </Pane>
            <Pane title="Retrieval results" icon={Radar} span={4}>
              <ul className="flex flex-col gap-1.5">
                {RETRIEVAL.map((row) => (
                  <li
                    key={row.id}
                    className="flex items-center gap-2 rounded-md border border-border/40 bg-background/40 px-2 py-1.5 text-xs"
                  >
                    <FileText className="size-3 shrink-0 text-muted-foreground" />
                    <span className="truncate text-foreground/80">{row.title}</span>
                    <span className="ml-auto inline-flex items-center gap-1 text-muted-foreground">
                      <span className="font-mono">{row.score.toFixed(2)}</span>
                      <Badge
                        variant="outline"
                        className={cn(
                          "h-5 px-1.5 py-0 text-[10px] uppercase tracking-wider",
                          row.status === "allow" && "bg-emerald-500/15 text-emerald-500",
                          row.status === "redact" && "bg-amber-500/15 text-amber-500",
                          row.status === "deny" && "bg-rose-500/15 text-rose-500"
                        )}
                      >
                        {row.status}
                      </Badge>
                    </span>
                  </li>
                ))}
              </ul>
            </Pane>
            <Pane title="Audit trail" icon={Clock} span={2}>
              <ul className="flex flex-col gap-1.5">
                {AUDIT.slice(0, 3).map((row) => (
                  <li
                    key={row.id}
                    className="rounded-md border border-border/40 bg-background/40 px-2 py-1.5 text-[10px] leading-snug"
                  >
                    <div className="flex items-center gap-1 font-mono text-muted-foreground">
                      <span>{row.time}</span>
                      <span className="truncate text-foreground/80">{row.action}</span>
                    </div>
                    <div className="truncate text-muted-foreground/70">{row.actor}</div>
                  </li>
                ))}
              </ul>
            </Pane>
          </TabsContent>

          <TabsContent value="sources" className="m-0 p-4">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Kind</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {SOURCES.map((s) => (
                  <TableRow key={s.id}>
                    <TableCell className="font-mono text-xs">{s.name}</TableCell>
                    <TableCell>{s.kind}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={cn(
                          s.status === "ok" && "bg-emerald-500/15 text-emerald-500",
                          s.status === "warn" && "bg-amber-500/15 text-amber-500",
                          s.status === "down" && "bg-rose-500/15 text-rose-500"
                        )}
                      >
                        {s.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TabsContent>

          <TabsContent value="routing" className="m-0 p-4">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Source</TableHead>
                  <TableHead className="text-right">Weight</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ROUTING.map((r) => (
                  <TableRow key={r.id}>
                    <TableCell className="font-mono text-xs">{r.label}</TableCell>
                    <TableCell className="text-right font-mono">
                      {(r.weight * 100).toFixed(0)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TabsContent>

          <TabsContent value="retrieval" className="m-0 p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Document</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead className="text-right">Score</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {RETRIEVAL.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="font-medium">{row.title}</TableCell>
                    <TableCell className="text-muted-foreground">{row.source}</TableCell>
                    <TableCell className="text-right font-mono">{row.score.toFixed(2)}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={cn(
                          "uppercase tracking-wider",
                          row.status === "allow" && "bg-emerald-500/15 text-emerald-500",
                          row.status === "redact" && "bg-amber-500/15 text-amber-500",
                          row.status === "deny" && "bg-rose-500/15 text-rose-500"
                        )}
                      >
                        {row.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TabsContent>

          <TabsContent value="audit" className="m-0 p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Actor</TableHead>
                  <TableHead>Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {AUDIT.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="font-mono text-xs">{row.time}</TableCell>
                    <TableCell className="text-muted-foreground">{row.actor}</TableCell>
                    <TableCell>{row.action}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TabsContent>
        </Tabs>

        <Separator />

        <div className="flex flex-wrap items-center justify-between gap-3 bg-background/40 px-4 py-2 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <Database className="size-3" />
            workspace.db · 4.2 GB · 12 ms p95
          </span>
          <span className="inline-flex items-center gap-1.5">
            <GitBranch className="size-3" />
            graph · 8.4k entities · 12 communities
          </span>
          <span className="inline-flex items-center gap-1.5">
            <Layers className="size-3" />
            3 members · 2 roles
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

interface PaneProps {
  readonly title: string;
  readonly icon: React.ComponentType<{ className?: string }>;
  readonly span: 2 | 3 | 4;
  readonly children: React.ReactNode;
}

function Pane({ title, icon: Icon, span, children }: PaneProps) {
  const colSpan = {
    2: "col-span-12 md:col-span-3",
    3: "col-span-12 md:col-span-3",
    4: "col-span-12 md:col-span-3",
  } as const;
  return (
    <div className={cn("flex flex-col gap-3 bg-card/40 p-4", colSpan[span])}>
      <div className="flex items-center justify-between">
        <span className="text-eyebrow text-muted-foreground">{title}</span>
        <Icon className="size-3.5 text-muted-foreground" />
      </div>
      {children}
    </div>
  );
}