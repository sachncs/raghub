"use client";

import * as React from "react";
import Link from "next/link";
import { motion } from "motion/react";

import {
  ArrowRight,
  ArrowUpRight,
  Check,
  Lightning,
  Shield,
  Sparkles,
} from "@/lib/icons";

import { Button } from "@/components/ui/button";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { Wordmark } from "@/components/wordmark";

interface FeatureCard {
  readonly title: string;
  readonly description: string;
  readonly icon: React.ComponentType<{ className?: string }>;
  readonly accent: string;
}

const FEATURES: readonly FeatureCard[] = [
  {
    title: "Multi-agent retrieval",
    description:
      "Vector, keyword, graph, memory, and web agents fan out in parallel and fuse into one ranked answer.",
    icon: Sparkles,
    accent: "from-indigo-500/20 to-indigo-500/5",
  },
  {
    title: "Encrypted workspace",
    description:
      "Every workspace is a sealed SQLite file, opened only with your passphrase. No data leaves your host.",
    icon: Shield,
    accent: "from-emerald-500/20 to-emerald-500/5",
  },
  {
    title: "Per-user strategy",
    description:
      "Different roles see different corpora. Document ACLs, group memberships, and admin overrides in one rule.",
    icon: Lightning,
    accent: "from-amber-500/20 to-amber-500/5",
  },
];

const FAQ: readonly { q: string; a: string }[] = [
  {
    q: "Where does my data live?",
    a: "Locally. Each workspace is one encrypted SQLite file on your machine. Nothing is uploaded to a third-party service.",
  },
  {
    q: "Do you support my LLM provider?",
    a: "Yes — OpenAI, Anthropic, AWS Bedrock, LiteLLM proxies, and any OpenAI-compatible API.",
  },
  {
    q: "How do multi-agent retrievals work?",
    a: "A root RagAgent orchestrates a fan-out to vector, keyword, graph, memory, and web sub-agents, then fuses their results with RRF.",
  },
];

export default function MarketingHome() {
  return (
    <>
      <section className="container relative flex flex-col items-center justify-center px-6 pb-20 pt-24 text-center md:pt-32">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/50 px-3 py-1 text-xs font-medium text-muted-foreground backdrop-blur-sm">
          <span className="size-1.5 rounded-full bg-emerald-500" />
          Local-first · No cloud required
        </div>
        <h1 className="max-w-4xl text-balance text-5xl font-semibold leading-[1.05] tracking-tight md:text-7xl">
          Every retrieval,
          <br />
          <span className="text-gradient-brand">extracted.</span>
        </h1>
        <p className="mt-6 max-w-2xl text-pretty text-lg text-muted-foreground md:text-xl">
          Vector, keyword, graph, memory, and web — one engine. Revex is
          hybrid retrieval built for teams, on your terms.
        </p>

        <DemoSearch />

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Button asChild size="lg">
            <Link href="/onboarding">
              Create workspace
              <ArrowRight className="size-4" />
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link href="/sign-in">Sign in</Link>
          </Button>
        </div>
        <p className="mt-4 text-xs text-muted-foreground">
          Free, MIT-licensed, self-hosted.
        </p>
      </section>

      <section className="container px-6 pb-24">
        <div className="grid gap-4 md:grid-cols-3">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              viewport={{ once: true }}
              className="group relative overflow-hidden rounded-2xl border border-border/60 bg-card/60 p-6 backdrop-blur-sm transition-all hover:-translate-y-0.5 hover:border-border hover:bg-card"
            >
              <div
                className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${f.accent} opacity-50 transition-opacity group-hover:opacity-80`}
                aria-hidden
              />
              <div className="relative">
                <div className="mb-4 inline-flex items-center justify-center rounded-lg bg-background p-2 ring-1 ring-border/50">
                  <f.icon className="size-4" />
                </div>
                <h3 className="mb-2 text-base font-semibold">{f.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {f.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="container px-6 pb-24">
        <div className="rounded-3xl border border-border/60 bg-card/40 p-8 backdrop-blur-sm md:p-12">
          <div className="grid gap-12 md:grid-cols-2 md:items-center">
            <div>
              <h2 className="text-balance text-3xl font-semibold leading-tight tracking-tight md:text-4xl">
                Three retrieval patterns.
                <br />
                One façade.
              </h2>
              <p className="mt-4 text-pretty text-muted-foreground">
                Revex orchestrates Graph, Swarm, and Workflow patterns behind
                one API. Per-user strategies let you tune how each user
                retrieves — without changing your code.
              </p>
              <ul className="mt-6 space-y-2">
                {[
                  "Dense + BM25 fused with RRF (k=60)",
                  "Parallel sub-agent fan-out",
                  "Session memory with auto-summarization",
                  "Per-document ACLs (user / role / group)",
                ].map((line) => (
                  <li key={line} className="flex items-start gap-2 text-sm">
                    <Check className="mt-0.5 size-4 shrink-0 text-emerald-500" />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-8">
                <Button asChild variant="outline" size="sm">
                  <Link href="/onboarding">
                    Try it now
                    <ArrowUpRight className="size-4" />
                  </Link>
                </Button>
              </div>
            </div>
            <Architecture />
          </div>
        </div>
      </section>

      <section className="container px-6 pb-24">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
            Questions, answered.
          </h2>
        </div>
        <div className="mx-auto mt-10 max-w-2xl divide-y divide-border/60 rounded-2xl border border-border/60 bg-card/40 backdrop-blur-sm">
          {FAQ.map((item) => (
            <details key={item.q} className="group p-6">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-sm font-medium">
                {item.q}
                <ArrowRight className="size-4 rotate-90 transition-transform group-open:rotate-180" />
              </summary>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                {item.a}
              </p>
            </details>
          ))}
        </div>
      </section>

      <section className="container px-6 pb-24 text-center">
        <div className="rounded-3xl border border-border/60 bg-gradient-to-br from-indigo-500/10 via-card to-amber-500/10 p-12 backdrop-blur-sm">
          <Wordmark size="lg" className="justify-center" />
          <h2 className="mt-4 text-balance text-3xl font-semibold tracking-tight md:text-4xl">
            Try Revex in 60 seconds.
          </h2>
          <p className="mx-auto mt-3 max-w-md text-pretty text-muted-foreground">
            Name your workspace, set an admin, and pick your LLM provider.
            Your encrypted workspace is ready before you finish your coffee.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <Button asChild size="lg">
              <Link href="/onboarding">
                Create workspace
                <ArrowRight className="size-4" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="https://github.com/sachncs/revex" target="_blank">
                Read the code
                <ArrowUpRight className="size-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </>
  );
}

function DemoSearch() {
  const [value, setValue] = React.useState("");
  const [submitted, setSubmitted] = React.useState<string | null>(null);
  const [thinking, setThinking] = React.useState(false);
  const answers: Readonly<Record<string, string>> = {
    default:
      "Revex orchestrates vector, keyword, graph, memory, and web agents in parallel. Try asking: 'How does hybrid retrieval work?'",
    hybrid:
      "Hybrid retrieval fuses dense vector scores with BM25 keyword scores using Reciprocal Rank Fusion (RRF, k=60). Each source contributes rank, not just raw score.",
    workspace:
      "A workspace is one sealed SQLite file, opened only with your passphrase. Documents, embeddings, ACLs, and audit trail live inside it.",
    rag:
      "RAG = Retrieval-Augmented Generation. Instead of asking an LLM to remember everything, we retrieve the relevant chunks and ground the answer in them.",
  };
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = value.trim();
    if (!q) return;
    setSubmitted(q);
    setThinking(true);
    setTimeout(() => setThinking(false), 700);
  };
  const lower = (submitted ?? "").toLowerCase();
  const response =
    lower.includes("hybrid") ? answers.hybrid :
    lower.includes("workspace") ? answers.workspace :
    lower.includes("rag") ? answers.rag :
    submitted ? answers.default : null;

  return (
    <div className="mt-10 w-full max-w-2xl">
      <form
        onSubmit={handleSubmit}
        className="group relative flex items-center gap-2 rounded-2xl border border-border/60 bg-card/70 p-2 shadow-sm backdrop-blur-sm transition-all focus-within:border-primary/40 focus-within:shadow-md"
      >
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask anything about Revex…"
          aria-label="Demo query"
          className="flex-1 bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground"
        />
        <KbdGroup>
          <Kbd>⌘</Kbd>
          <Kbd>K</Kbd>
        </KbdGroup>
        <Button type="submit" size="sm" className="rounded-xl">
          {thinking ? (
            <span className="size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : (
            <ArrowRight className="size-4" />
          )}
        </Button>
      </form>
      {response && !thinking && (
        <div className="mt-3 rounded-xl border border-border/60 bg-card/60 p-4 text-left text-sm text-muted-foreground backdrop-blur-sm">
          <span className="mb-1 block text-xs font-medium uppercase tracking-wider text-foreground">
            Demo response
          </span>
          {response}
        </div>
      )}
      <p className="mt-2 text-center text-xs text-muted-foreground">
        Demo only · real responses stream in <Link href="/chat" className="underline underline-offset-4">chat</Link>.
      </p>
    </div>
  );
}

function Architecture() {
  return (
    <div className="relative">
      <div className="rounded-2xl border border-border/60 bg-background/70 p-4 font-mono text-xs shadow-sm backdrop-blur-sm md:p-6 md:text-sm">
        <pre className="overflow-x-auto leading-relaxed">
          <code>{`┌──────────────────────────────────────────┐
│              RagAgent (root)             │
├──────────────────────────────────────────┤
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │
│   │ vec  │ │ bm25 │ │graph │ │ web  │    │
│   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘    │
│      └────────┴───────┴────────┘         │
│              RRF (k=60)                   │
│                 │                         │
│             generator                     │
└──────────────────────────────────────────┘`}</code>
        </pre>
      </div>
    </div>
  );
}