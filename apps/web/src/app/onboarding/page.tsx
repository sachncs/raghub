'use client';

import { useMemo, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

type Step = 'workspace' | 'admin' | 'llm' | 'passphrase' | 'confirm';

interface OnboardingState {
  workspaceName: string;
  adminEmail: string;
  adminPassword: string;
  llmProvider: 'openai' | 'minimax' | 'litellm' | 'anthropic' | 'bedrock';
  llmModel: string;
  llmApiKey: string;
  llmBaseUrl: string;
  passphrase: string;
  passphraseConfirm: string;
}

const STEPS: readonly Step[] = ['workspace', 'admin', 'llm', 'passphrase', 'confirm'];

const defaultState = (): OnboardingState => ({
  workspaceName: '',
  adminEmail: '',
  adminPassword: '',
  llmProvider: 'openai',
  llmModel: 'gpt-4.1',
  llmApiKey: '',
  llmBaseUrl: '',
  passphrase: '',
  passphraseConfirm: '',
});

const providerDefaultModel = (p: OnboardingState['llmProvider']): string => {
  switch (p) {
    case 'openai':
      return 'gpt-4.1';
    case 'minimax':
      return 'MiniMax-Text-01';
    case 'litellm':
      return 'gpt-4.1';
    case 'anthropic':
      return 'claude-3-5-sonnet-latest';
    case 'bedrock':
      return 'anthropic.claude-3-5-sonnet-20241022-v2:0';
  }
};

const providerDefaultBaseUrl = (p: OnboardingState['llmProvider']): string => {
  switch (p) {
    case 'openai':
      return '';
    case 'minimax':
      return 'https://api.minimax.chat/v1';
    case 'litellm':
      return 'http://localhost:4000/v1';
    case 'anthropic':
      return '';
    case 'bedrock':
      return '';
  }
};

export default function OnboardingPage() {
  const [step, setStep] = useState<Step>('workspace');
  const [state, setState] = useState<OnboardingState>(defaultState());
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const stepIndex = useMemo(() => STEPS.indexOf(step), [step]);

  const update = (patch: Partial<OnboardingState>): void => {
    setState((prev) => ({ ...prev, ...patch }));
  };

  const goNext = (): void => {
    const i = STEPS.indexOf(step);
    if (i < STEPS.length - 1) {
      const next = STEPS[i + 1];
      if (next) setStep(next);
    }
  };

  const goPrev = (): void => {
    const i = STEPS.indexOf(step);
    if (i > 0) {
      const prev = STEPS[i - 1];
      if (prev) setStep(prev);
    }
  };

  const submit = async (): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch('/api/proxy', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-raghub-path': '/v1/auth/register',
        },
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
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(body?.error?.message ?? 'register failed');
        return;
      }
      document.cookie = `raghub_token=${body.token}; path=/; max-age=86400; samesite=lax`;
      window.location.href = '/chat';
    } finally {
      setBusy(false);
    }
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

  return (
    <main className="container max-w-lg py-12">
      <h1 className="mb-1 text-2xl font-semibold">Welcome to raghub</h1>
      <p className="mb-6 text-sm text-muted-foreground">
        Step {stepIndex + 1} of {STEPS.length} — {step}
      </p>

      <div className="mb-6 h-1 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full bg-primary transition-all"
          style={{ width: `${((stepIndex + 1) / STEPS.length) * 100}%` }}
        />
      </div>

      <div className="rounded-lg border bg-card p-6 text-card-foreground">
        {step === 'workspace' && (
          <section className="space-y-4">
            <Label htmlFor="ws">Workspace name</Label>
            <Input
              id="ws"
              value={state.workspaceName}
              onChange={(e) => update({ workspaceName: e.target.value })}
              placeholder="Acme Research"
              autoFocus
            />
            <p className="text-xs text-muted-foreground">
              Each workspace keeps its own documents, groups, and audit trail.
            </p>
          </section>
        )}

        {step === 'admin' && (
          <section className="space-y-4">
            <Label htmlFor="email">Admin email</Label>
            <Input
              id="email"
              type="email"
              value={state.adminEmail}
              onChange={(e) => update({ adminEmail: e.target.value })}
              placeholder="you@company.com"
              autoFocus
            />
            <Label htmlFor="pw">Password</Label>
            <Input
              id="pw"
              type="password"
              value={state.adminPassword}
              onChange={(e) => update({ adminPassword: e.target.value })}
              placeholder="at least 8 characters"
            />
          </section>
        )}

        {step === 'llm' && (
          <section className="space-y-4">
            <Label htmlFor="prov">Provider</Label>
            <select
              id="prov"
              className="block w-full rounded-md border bg-background px-3 py-2 text-sm"
              value={state.llmProvider}
              onChange={(e) => {
                const p = e.target.value as OnboardingState['llmProvider'];
                update({
                  llmProvider: p,
                  llmModel: providerDefaultModel(p),
                  llmBaseUrl: providerDefaultBaseUrl(p),
                });
              }}
            >
              <option value="openai">OpenAI</option>
              <option value="minimax">MiniMax</option>
              <option value="litellm">LiteLLM (local proxy)</option>
              <option value="anthropic">Anthropic</option>
              <option value="bedrock">AWS Bedrock</option>
            </select>
            <Label htmlFor="model">Model</Label>
            <Input
              id="model"
              value={state.llmModel}
              onChange={(e) => update({ llmModel: e.target.value })}
            />
            <Label htmlFor="apikey">API key</Label>
            <Input
              id="apikey"
              type="password"
              value={state.llmApiKey}
              onChange={(e) => update({ llmApiKey: e.target.value })}
              placeholder="sk-..."
            />
            <Label htmlFor="baseurl">Base URL (optional)</Label>
            <Input
              id="baseurl"
              value={state.llmBaseUrl}
              onChange={(e) => update({ llmBaseUrl: e.target.value })}
              placeholder={providerDefaultBaseUrl(state.llmProvider)}
            />
            <p className="text-xs text-muted-foreground">
              Stored in <code>workspace_settings</code>, encrypted with your passphrase.
            </p>
          </section>
        )}

        {step === 'passphrase' && (
          <section className="space-y-4">
            <Label htmlFor="pp">Workspace passphrase</Label>
            <Input
              id="pp"
              type="password"
              value={state.passphrase}
              onChange={(e) => update({ passphrase: e.target.value })}
              placeholder="at least 8 characters"
            />
            <Label htmlFor="pp2">Confirm passphrase</Label>
            <Input
              id="pp2"
              type="password"
              value={state.passphraseConfirm}
              onChange={(e) => update({ passphraseConfirm: e.target.value })}
            />
            <p className="text-xs text-muted-foreground">
              Unlocks the workspace on subsequent logins. Required because raghub
              stores API keys encrypted at rest. Lose it and the keys are lost
              too — keep a copy in your password manager.
            </p>
          </section>
        )}

        {step === 'confirm' && (
          <section className="space-y-3 text-sm">
            <h2 className="text-base font-semibold">Review</h2>
            <Row label="Workspace" value={state.workspaceName} />
            <Row label="Admin" value={state.adminEmail} />
            <Row
              label="LLM"
              value={`${state.llmProvider} / ${state.llmModel}`}
            />
            <Row label="Passphrase" value="••••••••" />
          </section>
        )}

        {error && <p className="mt-4 text-sm text-destructive">{error}</p>}

        <div className="mt-6 flex justify-between">
          <Button variant="outline" onClick={goPrev} disabled={stepIndex === 0 || busy}>
            Back
          </Button>
          {step !== 'confirm' ? (
            <Button onClick={goNext} disabled={!canAdvance(step) || busy}>
              Next
            </Button>
          ) : (
            <Button onClick={submit} disabled={busy}>
              {busy ? 'Creating…' : 'Create workspace'}
            </Button>
          )}
        </div>
      </div>
    </main>
  );
}

const Row = ({ label, value }: { label: string; value: string }): JSX.Element => (
  <div className="flex justify-between border-b border-border py-2 last:border-0">
    <span className="text-muted-foreground">{label}</span>
    <span className="font-mono text-xs">{value}</span>
  </div>
);