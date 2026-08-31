"use client";

import * as React from "react";
import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function PrivacyPage() {
  return (
    <div className="container max-w-3xl px-6 py-20">
      <Link
        href="/"
        className="mb-6 inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
      >
        ← Back
      </Link>
      <h1 className="mb-2 text-3xl font-semibold tracking-tight">Privacy</h1>
      <p className="mb-10 text-sm text-muted-foreground">
        Last updated: today. Revex is local-first; we don&apos;t collect data
        about your workspace.
      </p>

      <div className="prose prose-sm max-w-none space-y-6 text-foreground">
        <Section title="1. What Revex collects">
          <p>
            Revex is a local-first application. The web UI runs in your
            browser, the API runs on your machine, and your workspace is one
            encrypted SQLite file on your disk. We do not have a hosted
            service that sees your data.
          </p>
        </Section>
        <Section title="2. Cookies">
          <p>
            The web UI sets two cookies on sign-in: <code>revex_session</code>{" "}
            (a short-lived JWT) and <code>revex_workspace_key</code> (your
            workspace passphrase, used to unlock the encrypted workspace).
            Both are required for the app to function.
          </p>
        </Section>
        <Section title="3. LLM provider data">
          <p>
            When you configure an LLM provider, your prompts and the
            retrieved document chunks are sent to that provider&apos;s API.
            Revex does not proxy these requests — they go directly from
            your machine to the provider you configured.
          </p>
        </Section>
        <Section title="4. Telemetry">
          <p>
            Telemetry is opt-in. The default is <code>noop</code>. If you
            configure Langfuse or OpenTelemetry, traces are sent to the
            endpoint you provide.
          </p>
        </Section>
        <Section title="5. Your rights">
          <p>
            You own your workspace file. Deleting it deletes your data. There
            is no account to deactivate, because there is no account on our
            servers.
          </p>
        </Section>
      </div>

      <div className="mt-12 flex items-center gap-2">
        <Button asChild variant="outline">
          <Link href="/terms">Read terms</Link>
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