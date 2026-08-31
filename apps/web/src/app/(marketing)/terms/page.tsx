"use client";

import * as React from "react";
import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function TermsPage() {
  return (
    <div className="container max-w-3xl px-6 py-20">
      <Link
        href="/"
        className="mb-6 inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
      >
        ← Back
      </Link>
      <h1 className="mb-2 text-3xl font-semibold tracking-tight">Terms</h1>
      <p className="mb-10 text-sm text-muted-foreground">
        Revex is MIT-licensed open-source software. By using it, you agree
        to the following.
      </p>

      <div className="prose prose-sm max-w-none space-y-6 text-foreground">
        <Section title="1. License">
          <p>
            Revex is released under the MIT License. See the{" "}
            <code>LICENSE</code> file in the repository for the full text.
          </p>
        </Section>
        <Section title="2. No warranty">
          <p>
            The software is provided &quot;as is&quot;, without warranty of any kind,
            express or implied. See the MIT License for the full disclaimer.
          </p>
        </Section>
        <Section title="3. Your responsibility">
          <p>
            You are responsible for securing your workspace passphrase.
            Losing it means losing access to your encrypted data. There is no
            recovery mechanism because there is no central authority that can
            recover it for you.
          </p>
        </Section>
        <Section title="4. Acceptable use">
          <p>
            You will not use Revex to retrieve, store, or generate content
            that violates applicable laws or third-party rights, including
            intellectual property and privacy rights.
          </p>
        </Section>
        <Section title="5. Third-party services">
          <p>
            Revex can connect to third-party services (LLM providers,
            telemetry endpoints, web search APIs). Your use of those services
            is governed by their terms, not these.
          </p>
        </Section>
      </div>

      <div className="mt-12 flex items-center gap-2">
        <Button asChild variant="outline">
          <Link href="/privacy">Read privacy</Link>
        </Button>
        <Button asChild>
          <Link href="/onboarding">Create a workspace</Link>
        </Button>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h2 className="mb-2 text-lg font-semibold">{title}</h2>
      <div className="space-y-3 text-sm leading-relaxed text-muted-foreground [&_code]:rounded [&_code]:bg-muted [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-xs [&_code]:text-foreground">
        {children}
      </div>
    </section>
  );
}