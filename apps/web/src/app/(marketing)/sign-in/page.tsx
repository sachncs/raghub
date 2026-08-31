"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { motion } from "motion/react";
import { ArrowRight, Eye, EyeOff, Sparkles } from "@/lib/icons";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Wordmark } from "@/components/wordmark";

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
    if (passphrase.length < 1) next.passphrase = "Workspace passphrase is required.";
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
        toast.error(body?.error?.message ?? "Sign in failed");
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
    <div className="container grid min-h-[calc(100vh-4rem)] items-center gap-8 px-6 py-12 md:grid-cols-2 md:py-20">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="mx-auto w-full max-w-sm"
      >
        <div className="mb-6 flex items-center justify-between">
          <Wordmark size="md" />
          <span className="rounded-full border border-border/60 bg-card/60 px-2.5 py-0.5 text-xs text-muted-foreground backdrop-blur-sm">
            Sign in
          </span>
        </div>
        <h1 className="mb-2 text-3xl font-semibold tracking-tight">
          Welcome back
        </h1>
        <p className="mb-6 text-sm text-muted-foreground">
          Sign in to your Revex workspace.
        </p>
        <form onSubmit={submit} className="space-y-4">
          <div className="space-y-1.5">
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
          <div className="space-y-1.5">
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
          <div className="space-y-1.5">
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
                className="absolute right-2 top-1/2 inline-flex size-6 -translate-y-1/2 items-center justify-center rounded text-muted-foreground hover:text-foreground"
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
          <Button type="submit" disabled={submitting} className="w-full" size="lg">
            {submitting ? (
              <>
                <span className="size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Signing in…
              </>
            ) : (
              <>
                Sign in
                <ArrowRight className="size-4" />
              </>
            )}
          </Button>
          <p className="text-center text-xs text-muted-foreground">
            New here?{" "}
            <Link
              href="/onboarding"
              className="font-medium text-foreground underline-offset-4 hover:underline"
            >
              Create a workspace
            </Link>
          </p>
        </form>
      </motion.div>
      <motion.aside
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
        className="hidden md:block"
      >
        <div className="relative mx-auto aspect-square max-w-md overflow-hidden rounded-3xl border border-border/60 bg-card/40 p-8 backdrop-blur-sm">
          <div
            aria-hidden
            className="absolute inset-0 bg-gradient-to-br from-indigo-500/20 via-amber-500/10 to-transparent"
          />
          <div className="relative flex h-full flex-col">
            <div className="mb-6 inline-flex items-center gap-2 self-start rounded-full border border-border/60 bg-background/60 px-3 py-1 text-xs font-medium text-muted-foreground backdrop-blur-sm">
              <Sparkles className="size-3.5" />
              Encrypted workspace
            </div>
            <h2 className="text-balance text-3xl font-semibold leading-tight tracking-tight">
              Your team&apos;s knowledge,
              <br />
              <span className="text-gradient-brand">instantly retrievable.</span>
            </h2>
            <p className="mt-4 text-pretty text-sm text-muted-foreground">
              Revex fuses vector, keyword, graph, memory, and web into one
              ranked answer — backed by a sealed SQLite workspace that only
              opens with your passphrase.
            </p>
            <div className="mt-auto space-y-3 rounded-2xl border border-border/60 bg-background/60 p-4 font-mono text-xs">
              <div className="flex items-center gap-2">
                <span className="size-2 rounded-full bg-emerald-500" />
                <span className="text-foreground">workspace.db</span>
                <span className="ml-auto text-muted-foreground">sealed</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="size-2 rounded-full bg-indigo-500" />
                <span className="text-foreground">vector index</span>
                <span className="ml-auto text-muted-foreground">ready</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="size-2 rounded-full bg-amber-500" />
                <span className="text-foreground">acl graph</span>
                <span className="ml-auto text-muted-foreground">ready</span>
              </div>
            </div>
          </div>
        </div>
      </motion.aside>
    </div>
  );
}