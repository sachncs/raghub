"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import {
  ArrowRight,
  Eye,
  EyeOff,
  KeyRound,
  ScanSearch,
  ShieldCheck,
} from "@/lib/icons";
import { AuthCard } from "@/components/auth/auth-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LogoMark } from "@/components/marketing/logo-mark";

export default function SignInPage() {
  const router = useRouter();
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [passphrase, setPassphrase] = React.useState("");
  const [showPassphrase, setShowPassphrase] = React.useState(false);
  const [submitting, setSubmitting] = React.useState(false);
  const [errors, setErrors] = React.useState<{
    email?: string;
    password?: string;
    passphrase?: string;
  }>({});

  const validate = (): boolean => {
    const next: typeof errors = {};
    if (!email.includes("@")) next.email = "Enter a valid email address.";
    if (password.length < 1) next.password = "Password is required.";
    if (passphrase.length < 1)
      next.passphrase = "Workspace passphrase is required.";
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const submit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    if (!validate()) return;
    setSubmitting(true);
    try {
      const res = await fetch("/api/proxy", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-revex-path": "/v1/auth/login",
        },
        body: JSON.stringify({ email, password, passphrase }),
      });
      const body = (await res.json().catch(() => ({}))) as {
        token?: string;
        error?: { message?: string };
      };
      if (!res.ok || !body.token) {
        const message = body?.error?.message ?? "Sign in failed";
        toast.error(message, {
          description:
            res.status === 401
              ? "No account matches those credentials. Try onboarding to create a workspace."
              : res.status === 404
                ? "Workspace not found. Onboard to create one."
                : undefined,
          action:
            res.status === 401 || res.status === 404
              ? {
                  label: "Onboard",
                  onClick: () => router.push("/onboarding"),
                }
              : undefined,
        });
        return;
      }
      document.cookie = `revex_session=${body.token}; path=/; max-age=86400; samesite=lax`;
      document.cookie = `revex_workspace_key=${encodeURIComponent(passphrase)}; path=/; max-age=86400; samesite=lax`;
      toast.success("Signed in");
      router.push("/chat");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthCard
      left={
        <div className="flex h-full flex-col justify-between gap-10">
          <div className="flex items-center gap-2.5">
            <LogoMark size="md" />
            <span className="text-lg font-semibold italic tracking-tight">
              re
              <span className="not-italic text-primary">›</span>
              ex
            </span>
          </div>

          <div className="flex flex-col gap-5">
            <Badge variant="outline" className="w-fit gap-2">
              <ShieldCheck className="size-3.5 text-emerald-500" />
              Encrypted workspace · sealed SQLite
            </Badge>
            <h2 className="text-balance text-display-2 leading-tight">
              Your team&apos;s knowledge,
              <br />
              <span className="text-gradient-brand">policy-aware retrieval.</span>
            </h2>
            <p className="max-w-md text-pretty text-base leading-relaxed text-muted-foreground">
              Revex fuses vector, keyword, graph, memory, and web into one
              ranked answer — behind a workspace that only opens with your
              passphrase.
            </p>
            <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <ScanSearch className="size-4 text-primary" />
                9 retrieval sources, fused with RRF
              </li>
              <li className="flex items-center gap-2">
                <ShieldCheck className="size-4 text-emerald-500" />
                Document-level ACLs enforced at retrieval
              </li>
              <li className="flex items-center gap-2">
                <KeyRound className="size-4 text-amber-500" />
                Sealed SQLite file · your disk, your control
              </li>
            </ul>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/50 px-2.5 py-1">
              <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
              SOC 2 Type II
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/50 px-2.5 py-1">
              <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
              ISO 27001
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/50 px-2.5 py-1">
              <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
              HIPAA-ready
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/50 px-2.5 py-1">
              <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
              MIT core
            </span>
          </div>
        </div>
      }
      trustPills={["SOC 2 Type II", "ISO 27001", "HIPAA-ready"]}
    >
      <div className="flex flex-col gap-2">
        <Badge variant="outline" className="w-fit gap-2">
          <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
          Sign in
        </Badge>
        <h1 className="text-h1 text-foreground">Welcome back</h1>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Sign in to your Revex workspace. Three fields, no third-party auth,
          no tracking.
        </p>
      </div>

      <form onSubmit={submit} className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              if (errors.email) setErrors((p) => ({ ...p, email: undefined }));
            }}
            aria-invalid={!!errors.email}
            required
          />
          {errors.email && (
            <p className="text-xs text-destructive">{errors.email}</p>
          )}
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value);
              if (errors.password)
                setErrors((p) => ({ ...p, password: undefined }));
            }}
            aria-invalid={!!errors.password}
            required
          />
          {errors.password && (
            <p className="text-xs text-destructive">{errors.password}</p>
          )}
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="passphrase">Workspace passphrase</Label>
          <div className="relative">
            <Input
              id="passphrase"
              type={showPassphrase ? "text" : "password"}
              autoComplete="off"
              value={passphrase}
              onChange={(e) => {
                setPassphrase(e.target.value);
                if (errors.passphrase)
                  setErrors((p) => ({ ...p, passphrase: undefined }));
              }}
              aria-invalid={!!errors.passphrase}
              required
            />
            <button
              type="button"
              onClick={() => setShowPassphrase(!showPassphrase)}
              className="absolute right-2 top-1/2 inline-flex size-7 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground hover:bg-muted/40 hover:text-foreground"
              aria-label={showPassphrase ? "Hide passphrase" : "Show passphrase"}
            >
              {showPassphrase ? (
                <EyeOff className="size-4" />
              ) : (
                <Eye className="size-4" />
              )}
            </button>
          </div>
          {errors.passphrase && (
            <p className="text-xs text-destructive">{errors.passphrase}</p>
          )}
          <p className="text-xs text-muted-foreground">
            Required to unlock the encrypted workspace on disk.
          </p>
        </div>
        <Button
          type="submit"
          disabled={submitting}
          size="lg"
          className="mt-2 w-full rounded-xl"
        >
          {submitting ? (
            <span className="inline-flex items-center gap-2">
              <span className="size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              Unlocking workspace…
            </span>
          ) : (
            <>
              Sign in
              <ArrowRight className="size-4" />
            </>
          )}
        </Button>
      </form>

      <p className="text-center text-sm text-muted-foreground">
        New here?{" "}
        <Link
          href="/onboarding"
          className="font-medium text-foreground underline-offset-4 hover:underline"
        >
          Create a workspace
        </Link>
      </p>
    </AuthCard>
  );
}