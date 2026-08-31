"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "motion/react";
import { toast } from "sonner";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Eye,
  EyeOff,
  Shield,
  Sparkles,
  User as UserIcon,
  Key,
  Lightning,
} from "@/lib/icons";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Wordmark } from "@/components/wordmark";

type Step = 'workspace' | 'admin' | 'llm' | 'passphrase' | 'confirm';

interface OnboardingState {
  workspaceName: string;
  adminEmail: string;
  adminPassword: string;
  llmProvider: 'openai' | 'anthropic' | 'bedrock' | 'litellm' | 'minimax';
  llmModel: string;
  llmApiKey: string;
  llmBaseUrl: string;
  passphrase: string;
  passphraseConfirm: string;
}

const STEPS: readonly Step[] = ['workspace', 'admin', 'llm', 'passphrase', 'confirm'];

const STEP_META: Record<
  Step,
  { title: string; subtitle: string; icon: React.ComponentType<{ className?: string }> }
> = {
  workspace: {
    title: 'Name your workspace',
    subtitle:
      'Each workspace keeps its own documents, embeddings, members, and audit trail.',
    icon: Shield,
  },
  admin: {
    title: 'Set up the admin account',
    subtitle: 'You are the first user. Email + password are required.',
    icon: UserIcon,
  },
  llm: {
    title: 'Configure your LLM provider',
    subtitle: 'Revex stores API keys encrypted with your passphrase.',
    icon: Sparkles,
  },
  passphrase: {
    title: 'Choose a workspace passphrase',
    subtitle: 'Unlocks the encrypted workspace on subsequent logins.',
    icon: Key,
  },
  confirm: {
    title: 'Review and create',
    subtitle: 'Last look before we seal the workspace.',
    icon: Lightning,
  },
};

const PROVIDER_DEFAULTS: Record<
  OnboardingState['llmProvider'],
  { model: string; baseUrl: string }
> = {
  openai: { model: 'gpt-4.1', baseUrl: '' },
  anthropic: { model: 'claude-3-5-sonnet-latest', baseUrl: '' },
  bedrock: { model: 'anthropic.claude-3-5-sonnet-20241022-v2', baseUrl: '' },
  litellm: { model: 'gpt-4.1', baseUrl: 'http://localhost:4000/v1' },
  minimax: { model: 'MiniMax-Text-01', baseUrl: 'https://api.minimax.chat/v1' },
};

const defaultState = (): OnboardingState => ({
  workspaceName: '',
  adminEmail: '',
  adminPassword: '',
  llmProvider: 'openai',
  llmModel: PROVIDER_DEFAULTS.openai.model,
  llmApiKey: '',
  llmBaseUrl: '',
  passphrase: '',
  passphraseConfirm: '',
});

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = React.useState<Step>('workspace');
  const [state, setState] = React.useState<OnboardingState>(defaultState());
  const [busy, setBusy] = React.useState(false);
  const [showPassphrase, setShowPassphrase] = React.useState(false);
  const stepIndex = STEPS.indexOf(step);

  const update = (patch: Partial<OnboardingState>): void => {
    setState((prev) => ({ ...prev, ...patch }));
  };

  const goNext = (): void => {
    const next = STEPS[stepIndex + 1];
    if (next) setStep(next);
  };

  const goPrev = (): void => {
    const prev = STEPS[stepIndex - 1];
    if (prev) setStep(prev);
  };

  const canAdvance = (current: Step): boolean => {
    switch (current) {
      case 'workspace':
        return state.workspaceName.trim().length >= 2;
      case 'admin':
        return state.adminEmail.includes('@') && state.adminPassword.length >= 8;
      case 'llm':
        return state.llmModel.trim().length > 0 && state.llmApiKey.trim().length > 0;
      case 'passphrase':
        return (
          state.passphrase.length >= 8 &&
          state.passphrase === state.passphraseConfirm
        );
      case 'confirm':
        return true;
    }
  };

  const submit = async (): Promise<void> => {
    setBusy(true);
    try {
      const res = await fetch('/api/proxy', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-revex-path': '/v1/auth/register' },
        body: JSON.stringify({
          workspaceName: state.workspaceName,
          email: state.adminEmail,
          password: state.adminPassword,
          passphrase: state.passphrase,
          llm: {
            provider: state.llmProvider,
            model: state.llmModel,
            apiKey: state.llmApiKey || undefined,
            baseUrl: state.llmBaseUrl || undefined,
          },
        }),
      });
      const body = (await res.json().catch(() => ({}))) as {
        token?: string;
        error?: { message?: string };
      };
      if (!res.ok) {
        toast.error(body.error?.message ?? 'Workspace creation failed');
        return;
      }
      if (body.token) {
        document.cookie = `revex_session=${body.token}; path=/; max-age=86400; samesite=lax`;
      }
      if (state.passphrase.length > 0) {
        document.cookie = `revex_workspace_key=${encodeURIComponent(state.passphrase)}; path=/; max-age=86400; samesite=lax`;
      }
      toast.success('Workspace created');
      router.push('/chat');
    } finally {
      setBusy(false);
    }
  };

  const meta = STEP_META[step];
  const Icon = meta.icon;

  return (
    <div className="container max-w-3xl px-6 py-12 md:py-16">
      <div className="mb-8 flex items-center justify-between">
        <Link href="/" className="inline-flex">
          <Wordmark size="md" />
        </Link>
        <Link
          href="/sign-in"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Already have a workspace? <span className="underline underline-offset-4">Sign in</span>
        </Link>
      </div>

      <Stepper current={stepIndex} steps={STEPS} onJump={(idx) => {
        if (idx <= stepIndex) setStep(STEPS[idx] ?? 'workspace');
      }} />

      <div className="mt-8 rounded-2xl border border-border/60 bg-card/60 p-6 backdrop-blur-sm md:p-8">
        <div className="mb-6 flex items-start gap-3">
          <div className="rounded-lg bg-muted p-2 text-muted-foreground">
            <Icon className="size-5" />
          </div>
          <div>
            <h2 className="text-xl font-semibold tracking-tight">{meta.title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{meta.subtitle}</p>
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
          >
            {step === 'workspace' && (
              <div className="space-y-1.5">
                <Label htmlFor="ws">Workspace name</Label>
                <Input
                  id="ws"
                  value={state.workspaceName}
                  onChange={(e) => update({ workspaceName: e.target.value })}
                  placeholder="Acme Research"
                  autoFocus
                />
                <p className="text-xs text-muted-foreground">
                  Visible in the dashboard and audit trail.
                </p>
              </div>
            )}

            {step === 'admin' && (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="email">Admin email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={state.adminEmail}
                    onChange={(e) => update({ adminEmail: e.target.value })}
                    placeholder="you@company.com"
                    autoFocus
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="pw">Password</Label>
                  <Input
                    id="pw"
                    type="password"
                    value={state.adminPassword}
                    onChange={(e) => update({ adminPassword: e.target.value })}
                    placeholder="at least 8 characters"
                  />
                </div>
              </div>
            )}

            {step === 'llm' && (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="prov">Provider</Label>
                  <Select
                    value={state.llmProvider}
                    onValueChange={(v) => {
                      const p = v as OnboardingState['llmProvider'];
                      update({
                        llmProvider: p,
                        llmModel: PROVIDER_DEFAULTS[p].model,
                        llmBaseUrl: PROVIDER_DEFAULTS[p].baseUrl,
                      });
                    }}
                  >
                    <SelectTrigger id="prov">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="openai">OpenAI</SelectItem>
                      <SelectItem value="anthropic">Anthropic</SelectItem>
                      <SelectItem value="bedrock">AWS Bedrock</SelectItem>
                      <SelectItem value="litellm">LiteLLM (local proxy)</SelectItem>
                      <SelectItem value="minimax">minimax</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="model">Model</Label>
                  <Input
                    id="model"
                    value={state.llmModel}
                    onChange={(e) => update({ llmModel: e.target.value })}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="apikey">API key</Label>
                  <Input
                    id="apikey"
                    type="password"
                    value={state.llmApiKey}
                    onChange={(e) => update({ llmApiKey: e.target.value })}
                    placeholder="sk-..."
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="baseurl">Base URL (optional)</Label>
                  <Input
                    id="baseurl"
                    value={state.llmBaseUrl}
                    onChange={(e) => update({ llmBaseUrl: e.target.value })}
                    placeholder={PROVIDER_DEFAULTS[state.llmProvider].baseUrl}
                  />
                  <p className="text-xs text-muted-foreground">
                    Stored in workspace_settings, never logged.
                  </p>
                </div>
              </div>
            )}

            {step === 'passphrase' && (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="pp">Workspace passphrase</Label>
                  <div className="relative">
                    <Input
                      id="pp"
                      type={showPassphrase ? 'text' : 'password'}
                      value={state.passphrase}
                      onChange={(e) => update({ passphrase: e.target.value })}
                      placeholder="at least 8 characters"
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassphrase(!showPassphrase)}
                      className="absolute right-2 top-1/2 inline-flex size-6 -translate-y-1/2 items-center justify-center rounded text-muted-foreground hover:text-foreground"
                      aria-label={showPassphrase ? 'Hide passphrase' : 'Show passphrase'}
                    >
                      {showPassphrase ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                    </button>
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="pp2">Confirm passphrase</Label>
                  <Input
                    id="pp2"
                    type={showPassphrase ? 'text' : 'password'}
                    value={state.passphraseConfirm}
                    onChange={(e) => update({ passphraseConfirm: e.target.value })}
                  />
                  <p className="text-xs text-muted-foreground">
                    Lose it and the encrypted keys are lost too — keep a copy
                    in your password manager.
                  </p>
                </div>
              </div>
            )}

            {step === 'confirm' && (
              <dl className="divide-y divide-border/60 rounded-xl border border-border/60 bg-background/40">
                {[
                  { label: 'Workspace', value: state.workspaceName },
                  { label: 'Admin', value: state.adminEmail },
                  {
                    label: 'LLM',
                    value: `${state.llmProvider} / ${state.llmModel}`,
                  },
                  { label: 'Passphrase', value: '••••••••' },
                ].map((row) => (
                  <div
                    key={row.label}
                    className="flex items-center justify-between px-4 py-3 text-sm"
                  >
                    <dt className="text-muted-foreground">{row.label}</dt>
                    <dd className="font-mono text-xs">{row.value}</dd>
                  </div>
                ))}
              </dl>
            )}
          </motion.div>
        </AnimatePresence>

        <div className="mt-8 flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={goPrev}
            disabled={stepIndex === 0 || busy}
          >
            <ArrowLeft className="size-4" />
            Back
          </Button>
          {step !== 'confirm' ? (
            <Button onClick={goNext} disabled={!canAdvance(step) || busy}>
              Next
              <ArrowRight className="size-4" />
            </Button>
          ) : (
            <Button onClick={submit} disabled={busy} size="lg">
              {busy ? (
                <>
                  <span className="size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Creating…
                </>
              ) : (
                <>
                  Create workspace
                  <Check className="size-4" />
                </>
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function Stepper({
  current,
  steps,
  onJump,
}: {
  current: number;
  steps: readonly Step[];
  onJump: (idx: number) => void;
}) {
  return (
    <ol className="flex items-center gap-2">
      {steps.map((s, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li key={s} className="flex flex-1 items-center gap-2">
            <button
              type="button"
              onClick={() => onJump(i)}
              disabled={i > current}
              className="group flex items-center gap-2 disabled:cursor-not-allowed"
            >
              <span
                className={`flex size-7 items-center justify-center rounded-full text-xs font-medium transition-all ${
                  done
                    ? 'bg-primary text-primary-foreground'
                    : active
                      ? 'bg-foreground text-background ring-4 ring-foreground/10'
                      : 'bg-muted text-muted-foreground'
                }`}
              >
                {done ? <Check className="size-3.5" /> : i + 1}
              </span>
              <span
                className={`hidden text-xs font-medium uppercase tracking-wider sm:inline ${
                  active
                    ? 'text-foreground'
                    : done
                      ? 'text-muted-foreground'
                      : 'text-muted-foreground/60'
                }`}
              >
                {s}
              </span>
            </button>
            {i < steps.length - 1 && (
              <span
                className={`h-px flex-1 transition-colors ${
                  i < current ? 'bg-primary/60' : 'bg-border'
                }`}
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}