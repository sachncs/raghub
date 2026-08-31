import * as React from "react";
import Link from "next/link";

import { ArrowLeft, ArrowRight, ShieldCheck } from "@/lib/icons";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function PrivacyPage() {
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
            <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
            Privacy
          </Badge>
          <h1 className="text-h1 text-balance text-foreground">
            What Revex collects, and what it doesn&apos;t.
          </h1>
          <p className="mt-4 text-pretty text-base text-muted-foreground md:text-lg">
            A short, plain-English statement of how a local-first product
            handles data. No cookie banners. No third-party trackers.
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
                    <ShieldCheck className="size-3.5 text-emerald-500" />
                    <span className="text-eyebrow text-muted-foreground">
                      In short
                    </span>
                  </div>
                  <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
                    Revex is local-first. The web UI runs in your browser, the
                    API runs on your machine, and your workspace is one
                    encrypted SQLite file on your disk. We do not have a
                    hosted service that sees your data.
                  </p>
                </CardContent>
              </Card>
            </div>
          </aside>
          <div className="space-y-4">
            <PrivacySection title="1. What Revex collects">
              <p>
                Revex is a local-first application. The web UI runs in your
                browser, the API runs on your machine, and your workspace is
                one encrypted SQLite file on your disk. We do not have a
                hosted service that sees your data.
              </p>
            </PrivacySection>
            <PrivacySection title="2. Cookies">
              <p>
                The web UI sets two cookies on sign-in:{" "}
                <code>revex_session</code> (a short-lived JWT) and{" "}
                <code>revex_workspace_key</code> (your workspace passphrase,
                used to unlock the encrypted workspace). Both are required
                for the app to function.
              </p>
            </PrivacySection>
            <PrivacySection title="3. LLM provider data">
              <p>
                When you configure an LLM provider, your prompts and the
                retrieved document chunks are sent to that provider&apos;s API.
                Revex does not proxy these requests — they go directly from
                your machine to the provider you configured.
              </p>
            </PrivacySection>
            <PrivacySection title="4. Telemetry">
              <p>
                Telemetry is opt-in. The default is <code>noop</code>. If you
                configure Langfuse or OpenTelemetry, traces are sent to the
                endpoint you provide.
              </p>
            </PrivacySection>
            <PrivacySection title="5. Your rights">
              <p>
                You own your workspace file. Deleting it deletes your data.
                There is no account to deactivate, because there is no
                account on our servers.
              </p>
            </PrivacySection>
          </div>
        </div>

        <div className="mt-12 flex flex-wrap items-center gap-2">
          <Button asChild variant="outline">
            <Link href="/terms">Read terms</Link>
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

function PrivacySection({
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