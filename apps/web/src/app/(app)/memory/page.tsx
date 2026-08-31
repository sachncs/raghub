"use client";

import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

interface MemoryStats {
  readonly workspaceId: string;
  readonly documentCount: number;
  readonly chunkCount: number;
  readonly totalTokens: number;
  readonly embeddingBytes: number;
  readonly bytesOnDisk: number;
  readonly lastIngestedAt: number | null;
  readonly statusCounts: Readonly<Record<string, number>>;
  readonly sources: ReadonlyArray<{
    readonly id: string;
    readonly name: string;
    readonly kind: string;
    readonly chunkCount: number;
  }>;
  readonly capacity: {
    readonly perWorkspaceDocumentsSoftLimit: number;
    readonly perWorkspaceChunksSoftLimit: number;
    readonly note: string;
  };
}

interface VacuumResponse {
  readonly workspaceId: string;
  readonly vacuumedAt: number;
  readonly integrityBefore: unknown;
  readonly integrityAfter: unknown;
}

const formatBytes = (n: number): string => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(2)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
};

const formatRelative = (ms: number | null): string => {
  if (ms === null) return "never";
  const diff = Date.now() - ms;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.round(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.round(diff / 3_600_000)}h ago`;
  return `${Math.round(diff / 86_400_000)}d ago`;
};

export default function MemoryPage() {
  const [stats, setStats] = React.useState<MemoryStats | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [vacuuming, setVacuuming] = React.useState(false);
  const [vacuumedAt, setVacuumedAt] = React.useState<number | null>(null);

  const refresh = React.useCallback(async (): Promise<void> => {
    setLoading(true);
    try {
      const res = await fetch("/api/proxy", {
        headers: { "x-revex-path": "/v1/admin/stats" },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as MemoryStats;
      setStats(body);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void refresh();
  }, [refresh]);

  const vacuum = async (): Promise<void> => {
    setVacuuming(true);
    try {
      const res = await fetch("/api/proxy", {
        method: "POST",
        headers: { "x-revex-path": "/v1/admin/vacuum" },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as VacuumResponse;
      setVacuumedAt(body.vacuumedAt);
      void refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Vacuum failed");
    } finally {
      setVacuuming(false);
    }
  };

  if (loading && stats === null) {
    return (
      <div className="container max-w-5xl px-6 py-10">
        <div className="space-y-3">
          <div className="h-8 w-48 animate-pulse rounded-md bg-muted/40" />
          <div className="h-32 w-full animate-pulse rounded-2xl bg-muted/30" />
          <div className="h-48 w-full animate-pulse rounded-2xl bg-muted/30" />
        </div>
      </div>
    );
  }

  return (
    <div className="container max-w-5xl px-6 py-10 space-y-8">
      <header className="space-y-2">
        <Badge variant="outline" className="gap-2">
          <span className="size-1.5 rounded-full bg-indigo-500" aria-hidden />
          Memory & Index
        </Badge>
        <h1 className="text-h1 text-balance text-foreground">
          Workspace memory at a glance.
        </h1>
        <p className="text-pretty text-muted-foreground">
          Revex stores every workspace as one sealed SQLite file. This
          panel shows what is in it, what it costs, and what to do
          when it grows.
        </p>
      </header>

      {error && (
        <Card>
          <CardContent className="py-4 text-sm text-destructive">{error}</CardContent>
        </Card>
      )}

      {stats && (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <KpiCard
              label="Documents"
              value={stats.documentCount.toLocaleString()}
              hint={`soft limit ${stats.capacity.perWorkspaceDocumentsSoftLimit.toLocaleString()}`}
            />
            <KpiCard
              label="Chunks"
              value={stats.chunkCount.toLocaleString()}
              hint={`soft limit ${stats.capacity.perWorkspaceChunksSoftLimit.toLocaleString()}`}
            />
            <KpiCard
              label="Tokens"
              value={stats.totalTokens.toLocaleString()}
              hint="estimated via 4-char heuristic"
            />
            <KpiCard
              label="Embeddings"
              value={formatBytes(stats.embeddingBytes)}
              hint="vec0 virtual table bytes"
            />
          </section>

          <Card>
            <CardHeader>
              <CardTitle>Capacity</CardTitle>
              <CardDescription>{stats.capacity.note}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <CapacityBar
                label="Documents"
                used={stats.documentCount}
                limit={stats.capacity.perWorkspaceDocumentsSoftLimit}
              />
              <CapacityBar
                label="Chunks"
                used={stats.chunkCount}
                limit={stats.capacity.perWorkspaceChunksSoftLimit}
              />
              <Separator />
              <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
                <span>
                  Last indexed: <span className="font-mono text-foreground">{formatRelative(stats.lastIngestedAt)}</span>
                </span>
                {vacuumedAt && (
                  <span>
                    Vacuumed: <span className="font-mono text-foreground">{formatRelative(vacuumedAt)}</span>
                  </span>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Source breakdown</CardTitle>
              <CardDescription>
                Files you have uploaded, by MIME type and chunk count.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              {stats.sources.length === 0 ? (
                <p className="px-6 py-6 text-sm text-muted-foreground">
                  No sources ingested yet. Upload a document to populate this view.
                </p>
              ) : (
                <ul className="divide-y divide-border/40">
                  {stats.sources.map((s) => (
                    <li key={s.id} className="flex items-center justify-between gap-3 px-6 py-3 text-sm">
                      <div className="min-w-0">
                        <p className="truncate font-medium text-foreground">{s.name}</p>
                        <p className="font-mono text-xs text-muted-foreground">{s.kind}</p>
                      </div>
                      <Badge variant="outline">{s.chunkCount} chunks</Badge>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Maintenance</CardTitle>
              <CardDescription>
                Vacuum reclaims SQLite pages after large deletes. Run it
                after removing many documents.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-between gap-3">
              <p className="text-sm text-muted-foreground">
                Last vacuum: {vacuumedAt ? formatRelative(vacuumedAt) : "never"}
              </p>
              <div className="flex items-center gap-2">
                <Button variant="outline" onClick={() => void refresh()} disabled={loading}>
                  Refresh
                </Button>
                <Button onClick={() => void vacuum()} disabled={vacuuming}>
                  {vacuuming ? "Vacuuming…" : "Vacuum now"}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Index health</CardTitle>
              <CardDescription>
                Per-status document counts from the underlying store.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {Object.entries(stats.statusCounts).length === 0 ? (
                <p className="text-sm text-muted-foreground">No documents.</p>
              ) : (
                Object.entries(stats.statusCounts).map(([status, count]) => (
                  <Badge
                    key={status}
                    variant="outline"
                    className={
                      status === "failed"
                        ? "border-rose-500/40 bg-rose-500/10 text-rose-500"
                        : status === "indexing" || status === "pending"
                          ? "border-amber-500/40 bg-amber-500/10 text-amber-500"
                          : "border-emerald-500/30 bg-emerald-500/10 text-emerald-500"
                    }
                  >
                    {status} · {count}
                  </Badge>
                ))
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function KpiCard({
  label,
  value,
  hint,
}: {
  readonly label: string;
  readonly value: string;
  readonly hint: string;
}) {
  return (
    <Card>
      <CardContent className="space-y-1 p-5">
        <p className="text-eyebrow text-muted-foreground">{label}</p>
        <p className="text-display-2 font-semibold tracking-tight text-foreground">{value}</p>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

function CapacityBar({
  label,
  used,
  limit,
}: {
  readonly label: string;
  readonly used: number;
  readonly limit: number;
}) {
  const pct = limit > 0 ? Math.min(100, Math.round((used / limit) * 100)) : 0;
  const tone =
    pct >= 90 ? "bg-rose-500" : pct >= 70 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono">
          {used.toLocaleString()} / {limit.toLocaleString()} · {pct}%
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted/40">
        <div
          className={`h-full ${tone} transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}