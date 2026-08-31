import * as React from "react";
import Link from "next/link";

import { ArrowLeft, ArrowRight, Scale } from "@/lib/icons";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function TermsPage() {
  return (
    <>
      <section className="container px-6 pt-20 md:pt-28">
        <Link
          href="/"
          className="mb-6 inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="size-4" />
          Back to home
        </Link>
        <div className="max-w-3xl">
          <Badge variant="outline" className="mb-3 gap-2">
            <span className="size-1.5 rounded-full bg-indigo-500" aria-hidden />
            Terms
          </Badge>
          <h1 className="text-h1 text-balance text-foreground">
            MIT-licensed. Your machine. Your responsibility.
          </h1>
          <p className="mt-4 text-pretty text-base text-muted-foreground md:text-lg">
            Revex is open-source software. By running it, you accept these
            short, plain-English terms in addition to the MIT License.
          </p>
          <p className="mt-4 text-xs text-muted-foreground">
            Last updated: today.
          </p>
        </div>
      </section>

      <section className="container px-6 pb-24 pt-10">
        <div className="grid gap-8 md:grid-cols-[1fr_2fr]">
          <aside className="hidden md:block">
            <div className="sticky top-28">
              <Card>
                <CardContent className="space-y-3 p-5">
                  <div className="flex items-center gap-2">
                    <Scale className="size-3.5 text-primary" />
                    <span className="text-eyebrow text-muted-foreground">
                      In short
                    </span>
                  </div>
                  <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
                    Revex is MIT-licensed open-source software. You are
                    responsible for securing your workspace passphrase and
                    complying with third-party services you connect.
                  </p>
                </CardContent>
              </Card>
            </div>
          </aside>
          <div className="space-y-4">
            <TermSection title="1. License">
              <p>
                Revex is released under the MIT License. See the{" "}
                <code>LICENSE</code> file in the repository for the full
                text.
              </p>
            </TermSection>
            <TermSection title="2. No warranty">
              <p>
                The software is provided &quot;as is&quot;, without warranty
                of any kind, express or implied. See the MIT License for
                the full disclaimer.
              </p>
            </TermSection>
            <TermSection title="3. Your responsibility">
              <p>
                You are responsible for securing your workspace passphrase.
                Losing it means losing access to your encrypted data. There
                is no recovery mechanism because there is no central
                authority that can recover it for you.
              </p>
            </TermSection>
            <TermSection title="4. Acceptable use">
              <p>
                You will not use Revex to retrieve, store, or generate
                content that violates applicable laws or third-party rights,
                including intellectual property and privacy rights.
              </p>
            </TermSection>
            <TermSection title="5. Third-party services">
              <p>
                Revex can connect to third-party services (LLM providers,
                telemetry endpoints, web search APIs). Your use of those
                services is governed by their terms, not these.
              </p>
            </TermSection>
          </div>
        </div>

        <div className="mt-12 flex flex-wrap items-center gap-2">
          <Button asChild variant="outline">
            <Link href="/privacy">Read privacy</Link>
          </Button>
          <Button asChild>
            <Link href="/onboarding">
              Create a workspace
              <ArrowRight className="size-4" />
            </Link>
          </Button>
        </div>
      </section>
    </>
  );
}

function TermSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="bg-card/30">
      <CardContent className="space-y-3 p-6">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground [&_code]:rounded [&_code]:bg-muted [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-xs [&_code]:text-foreground">
          {children}
        </div>
      </CardContent>
    </Card>
  );
}