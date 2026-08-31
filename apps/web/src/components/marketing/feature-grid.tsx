import * as React from "react";

import {
  Brain,
  Database,
  FileText,
  GitBranch,
  Globe,
  Plug,
  Radar,
  ScanSearch,
  ShieldCheck,
} from "@/lib/icons";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

import { cn } from "@/lib/utils";

type FeatureIcon = React.ComponentType<{ className?: string }>;

interface Feature {
  readonly title: string;
  readonly description: string;
  readonly icon: FeatureIcon;
  readonly number: string;
}

const FEATURES: readonly Feature[] = [
  {
    title: "Dense vector retrieval",
    description:
      "OpenAI, Voyage, Cohere, Bedrock embeddings. HNSW + IVF indexes, auto-tuned.",
    icon: Database,
    number: "01",
  },
  {
    title: "BM25 keyword retrieval",
    description:
      "Lexical recall with custom analyzers, stopword policies, and per-corpus weighting.",
    icon: ScanSearch,
    number: "02",
  },
  {
    title: "Graph retrieval",
    description:
      "Entity-aware neighborhood traversal with community summaries and GraphRAG.",
    icon: GitBranch,
    number: "03",
  },
  {
    title: "Session memory",
    description:
      "Long-running context with auto-summarisation, TTL, and per-turn receipts.",
    icon: Brain,
    number: "04",
  },
  {
    title: "Web augmentation",
    description:
      "Optional grounded web search with strict source allowlists and freshness rules.",
    icon: Globe,
    number: "05",
  },
  {
    title: "Reranking",
    description:
      "Cohere, ColBERT MaxSim, LLM-as-judge, and your own scorers. Plug-in contract.",
    icon: Radar,
    number: "06",
  },
  {
    title: "Ingest anywhere",
    description:
      "Files, S3, Notion, Confluence, Slack, webhooks, schedulers. PDFs, DOCX, code, OCR.",
    icon: Plug,
    number: "07",
  },
  {
    title: "Policy-aware generation",
    description:
      "Generators refuse, redact, or annotate by retrieval scope. Citations are first-class.",
    icon: ShieldCheck,
    number: "08",
  },
];

interface FeatureGridProps {
  readonly className?: string;
}

export function FeatureGrid({ className }: FeatureGridProps) {
  return (
    <section
      id="capabilities"
      className={cn("relative isolate py-16 md:py-24", className)}
    >
      <div className="container px-6">
        <div className="mx-auto max-w-3xl text-center">
          <Badge variant="outline" className="mb-3 gap-2">
            <span className="size-1.5 rounded-full bg-indigo-500" aria-hidden />
            Capabilities
          </Badge>
          <h2 className="text-h1 text-balance text-foreground">
            Every retrieval, every source — one engine.
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-pretty text-base text-muted-foreground md:text-lg">
            Revex fuses the modalities teams actually need to answer real
            questions, not the ones demos love to show.
          </p>
        </div>

        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((feature) => (
            <Card
              key={feature.title}
              className="group transition-all hover:-translate-y-0.5 hover:bg-card/80 hover:shadow-lg hover:shadow-black/5"
            >
              <CardContent className="flex flex-col gap-3 p-5">
                <div className="flex items-center justify-between">
                  <span className="inline-flex size-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-500">
                    <feature.icon className="size-5" />
                  </span>
                  <span className="text-eyebrow text-muted-foreground/60">
                    {feature.number}
                  </span>
                </div>
                <h3 className="text-base font-semibold text-foreground">
                  {feature.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-8 grid gap-3 sm:grid-cols-3">
          <Card className="bg-card/30">
            <CardContent className="flex items-center gap-3 p-4 text-xs">
              <FileText className="size-4 text-indigo-500" />
              <span className="text-muted-foreground">
                <span className="text-foreground">Ingest → Govern → Retrieve → Answer → Audit.</span>{" "}
                Five steps, one pipeline.
              </span>
            </CardContent>
          </Card>
          <Card className="bg-card/30">
            <CardContent className="flex items-center gap-3 p-4 text-xs">
              <Brain className="size-4 text-indigo-500" />
              <span className="text-muted-foreground">
                <span className="text-foreground">Knowledge base, copilots, support, research, compliance.</span>{" "}
                One engine, six surfaces.
              </span>
            </CardContent>
          </Card>
          <Card className="bg-card/30">
            <CardContent className="flex items-center gap-3 p-4 text-xs">
              <ShieldCheck className="size-4 text-indigo-500" />
              <span className="text-muted-foreground">
                <span className="text-foreground">Audit trail and retention</span> built in, not bolted on.
              </span>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}