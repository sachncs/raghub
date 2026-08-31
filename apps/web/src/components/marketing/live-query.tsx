"use client";

import * as React from "react";
import { motion, AnimatePresence } from "motion/react";

import {
  ArrowRight,
  CircleNotch,
  ScanSearch,
  ShieldCheck,
  Sparkles,
} from "@/lib/icons";
import { useRevexStream } from "@/lib/hooks/use-revex-stream";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { cn } from "@/lib/utils";

interface LiveQueryProps {
  readonly className?: string;
  readonly path?: string;
}

interface TraceStep {
  readonly id: string;
  readonly label: string;
  readonly detail: string;
  readonly accent: "indigo" | "amber" | "emerald";
}

interface DemoAnswer {
  readonly question: string;
  readonly answer: string;
  readonly trace: readonly TraceStep[];
}

const DEMO_ANSWERS: readonly DemoAnswer[] = [
  {
    question: "How does hybrid retrieval work?",
    answer:
      "Hybrid retrieval fuses dense vector scores with BM25 keyword scores using Reciprocal Rank Fusion (RRF, k=60). Each source contributes rank, not raw score — so semantic match and lexical match combine into a single, defensible ranking.",
    trace: [
      { id: "vec", label: "Vector search", detail: "12 hits · 9.4 ms", accent: "indigo" },
      { id: "kw", label: "BM25 keyword", detail: "8 hits · 1.2 ms", accent: "indigo" },
      { id: "acl", label: "ACL gate", detail: "12 / 12 allowed · 0 redacted", accent: "emerald" },
      { id: "rrf", label: "RRF fusion", detail: "k = 60 · 9 unique", accent: "amber" },
    ],
  },
  {
    question: "How are document ACLs enforced?",
    answer:
      "Every retrieval is gated by the requester's role, group memberships, and per-document policies before any result is ranked. Revex refuses, redacts, or annotates based on the policy decision — never returns a document the user cannot access.",
    trace: [
      { id: "auth", label: "Resolver", detail: "user:jane · role:legal · 4 groups", accent: "indigo" },
      { id: "acl", label: "ACL gate", detail: "23 / 41 allowed · 18 deny", accent: "emerald" },
      { id: "redact", label: "Redaction", detail: "3 fields stripped", accent: "emerald" },
      { id: "rank", label: "Ranking", detail: "9 in scope", accent: "amber" },
    ],
  },
  {
    question: "Where does my data live?",
    answer:
      "Locally. Each workspace is one encrypted SQLite file on disk. Documents, embeddings, ACLs, and audit trail all live inside it. Nothing is uploaded to a third-party service, and the workspace is unusable without your passphrase.",
    trace: [
      { id: "ws", label: "Workspace", detail: "~/.revex/workspaces/prod.db", accent: "indigo" },
      { id: "enc", label: "Encryption", detail: "AES-256-GCM · sealed", accent: "emerald" },
      { id: "audit", label: "Audit trail", detail: "append-only ledger", accent: "amber" },
    ],
  },
];

const ACCENT_BG: Record<TraceStep["accent"], string> = {
  indigo: "bg-indigo-500",
  amber: "bg-amber-500",
  emerald: "bg-emerald-500",
};

export function LiveQuery({ className, path = "/v1/query/stream" }: LiveQueryProps) {
  const [value, setValue] = React.useState("");
  const [activeDemo, setActiveDemo] = React.useState<DemoAnswer | null>(null);
  const [streamingAnswer, setStreamingAnswer] = React.useState("");
  const [streaming, setStreaming] = React.useState(false);
  const stream = useRevexStream({ path });
  const [trace, setTrace] = React.useState<readonly TraceStep[]>([]);
  const [latency, setLatency] = React.useState<number | null>(null);
  const startedAt = React.useRef<number | null>(null);

  const submit = React.useCallback(
    (question: string) => {
      const q = question.trim();
      if (!q || streaming) return;
      setValue("");
      setActiveDemo(null);
      setStreamingAnswer("");
      setTrace([]);
      setLatency(null);
      startedAt.current = performance.now();
      setStreaming(true);
      const sessionId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `s_${Date.now().toString(36)}`;

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1200);

      fetch("/api/proxy", {
        method: "POST",
        headers: { "content-type": "application/json", "x-revex-path": path },
        body: JSON.stringify({ question: q, sessionId }),
        signal: controller.signal,
      })
        .then(async (res) => {
          clearTimeout(timeout);
          if (!res.ok || !res.body) throw new Error(`status_${res.status}`);
          await stream.start({ question: q, sessionId });
        })
        .catch(() => {
          clearTimeout(timeout);
          const match =
            DEMO_ANSWERS.find((d) =>
              q.toLowerCase().includes(d.question.toLowerCase().split(" ")[0] ?? "")
            ) ?? DEMO_ANSWERS[0];
          if (!match) return;
          setActiveDemo(match);
          let i = 0;
          const id = setInterval(() => {
            i += 1;
            setStreamingAnswer(match.answer.slice(0, i));
            if (i >= match.answer.length) {
              clearInterval(id);
              const ms = startedAt.current ? Math.round(performance.now() - startedAt.current) : 0;
              setLatency(ms);
              setStreaming(false);
            }
          }, 14);
        });
    },
    [path, stream, streaming]
  );

  React.useEffect(() => {
    if (stream.text && !streamingAnswer) {
      setStreamingAnswer(stream.text);
    }
    if (!stream.streaming && startedAt.current && streaming) {
      const ms = Math.round(performance.now() - startedAt.current);
      setLatency(ms);
      setStreaming(false);
    }
    if (stream.events.length > 0) {
      const derived = stream.events.slice(0, 6).map((e, i): TraceStep => ({
        id: `e-${i}`,
        label: e.kind,
        detail:
          typeof e.payload["detail"] === "string"
            ? (e.payload["detail"] as string)
            : "ok",
        accent: i % 3 === 0 ? "indigo" : i % 3 === 1 ? "amber" : "emerald",
      }));
      setTrace(derived);
    }
  }, [stream.events, stream.text, stream.streaming, streaming, streamingAnswer]);

  const visibleTrace = trace.length > 0 ? trace : (activeDemo?.trace ?? []);
  const visibleAnswer = streamingAnswer || stream.text;

  return (
    <div className={cn("w-full", className)}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
        className="group relative flex items-center gap-2 rounded-2xl border border-border/60 bg-card/70 p-2 shadow-sm shadow-black/5 backdrop-blur-md transition-all focus-within:border-primary/40 focus-within:shadow-lg focus-within:shadow-primary/10"
      >
        <span className="inline-flex size-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
          {streaming ? (
            <CircleNotch className="size-4 animate-spin" />
          ) : (
            <ScanSearch className="size-4" />
          )}
        </span>
        <Input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask anything about Revex…"
          aria-label="Demo query"
          className="flex-1 border-0 bg-transparent px-1 py-2 text-sm shadow-none outline-none placeholder:text-muted-foreground focus-visible:ring-0"
          disabled={streaming}
        />
        <KbdGroup className="hidden md:inline-flex">
          <Kbd>⌘</Kbd>
          <Kbd>K</Kbd>
        </KbdGroup>
        <Button type="submit" size="sm" className="rounded-xl" disabled={streaming}>
          {streaming ? (
            <CircleNotch className="size-4 animate-spin" />
          ) : (
            <>
              <span className="hidden sm:inline">Ask</span>
              <ArrowRight className="size-4" />
            </>
          )}
        </Button>
      </form>

      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        {DEMO_ANSWERS.map((demo) => (
          <button
            key={demo.question}
            type="button"
            onClick={() => submit(demo.question)}
            disabled={streaming}
            className="group flex items-start gap-2 rounded-xl border border-border/40 bg-card/30 p-3 text-left text-xs text-muted-foreground transition-all hover:border-border hover:bg-card/60 hover:text-foreground disabled:opacity-50"
          >
            <Sparkles className="mt-0.5 size-3.5 shrink-0 text-primary" />
            <span className="line-clamp-2 leading-snug">{demo.question}</span>
          </button>
        ))}
      </div>

      <AnimatePresence>
        {(visibleAnswer || streaming) && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="mt-4"
          >
            <Card>
              <CardContent className="space-y-3 p-5">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline" className="gap-1.5">
                    <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
                    Live trace
                  </Badge>
                  {latency !== null && (
                    <Badge variant="outline" className="gap-1.5">
                      <span className="size-1.5 rounded-full bg-amber-500" aria-hidden />
                      {latency} ms
                    </Badge>
                  )}
                </div>

                {visibleTrace.length > 0 && (
                  <Tabs defaultValue={visibleTrace[0]?.id ?? "trace"} className="w-full">
                    <TabsList className="h-auto flex-wrap justify-start gap-1 bg-transparent p-0">
                      {visibleTrace.map((step) => (
                        <TabsTrigger
                          key={step.id}
                          value={step.id}
                          className="gap-2 data-[state=active]:bg-background"
                        >
                          <span className={cn("size-1.5 rounded-full", ACCENT_BG[step.accent])} aria-hidden />
                          <span>{step.label}</span>
                          <span className="text-muted-foreground">{step.detail}</span>
                        </TabsTrigger>
                      ))}
                    </TabsList>
                    {visibleTrace.map((step) => (
                      <TabsContent key={step.id} value={step.id} className="mt-2">
                        <div className="rounded-lg border border-border/40 bg-background/40 px-3 py-2 text-xs text-muted-foreground">
                          <span className="font-medium text-foreground">{step.label}</span>
                          <span className="ml-2">{step.detail}</span>
                        </div>
                      </TabsContent>
                    ))}
                  </Tabs>
                )}

                <div className="text-pretty text-sm leading-relaxed text-foreground md:text-base">
                  {visibleAnswer || (
                    <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                      <CircleNotch className="size-3.5 animate-spin" />
                      Retrieving across 5 sources…
                    </span>
                  )}
                  {streaming && visibleAnswer && (
                    <span
                      className="ml-0.5 inline-block size-1.5 animate-pulse rounded-full bg-primary align-middle"
                      aria-hidden
                    />
                  )}
                </div>

                <div className="flex items-center justify-between gap-3 border-t border-border/40 pt-3 text-xs text-muted-foreground">
                  <span className="inline-flex items-center gap-1.5">
                    <ShieldCheck className="size-3.5" />
                    Policy-scoped · redacted
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    Demo response
                  </span>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}