'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldDescription, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

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
      return 'anthropic.claude-3-5-sonnet-20241022-v2';
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
  const progressPct = ((stepIndex + 1) / STEPS.length) * 100;

  const update = (patch: Partial<OnboardingState>): void => {
    setState((prev) => ({ ...prev, ...patch }));
  };

  const goNext = (): void => {
    const i = STEPS.indexOf(step);
    const next = STEPS[i + 1];
    if (next) setStep(next);
  };

  const goPrev = (): void => {
    const i = STEPS.indexOf(step);
    const prev = STEPS[i - 1];
    if (prev) setStep(prev);
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
        const message = (body as { error?: { message?: string } })?.error?.message ?? 'register failed';
        setError(message);
        toast.error(message);
        return;
      }
      const token = (body as { token?: string }).token;
      if (token) {
        document.cookie = `raghub_token=${token}; path=/; max-age=86400; samesite=lax`;
      }
      toast.success('Workspace created');
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
    <div className="container max-w-lg py-12">
      <h1 className="mb-1 text-2xl font-semibold">Welcome to raghub</h1>
      <p className="mb-6 text-sm text-muted-foreground">
        Step {stepIndex + 1} of {STEPS.length} — {step}
      </p>

      <div
        className="mb-6 h-1 w-full overflow-hidden rounded-full bg-secondary"
        role="progressbar"
        aria-label={`Step ${stepIndex + 1} of ${STEPS.length}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progressPct}
      >
        <div
          className="h-full bg-primary transition-all"
          style={{ width: `${progressPct}%` }}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{step === 'workspace' ? 'Name your workspace' :
              step === 'admin' ? 'Set up the admin account' :
              step === 'llm' ? 'Configure your LLM provider' :
              step === 'passphrase' ? 'Choose a workspace passphrase' :
              'Review and create'}</CardTitle>
          <CardDescription>
            {step === 'workspace' ? 'Each workspace keeps its own documents, groups, and audit trail.' :
              step === 'admin' ? 'You are the first user. Email + password are required.' :
              step === 'llm' ? 'raghub stores API keys encrypted with your passphrase.' :
              step === 'passphrase' ? 'Unlocks the workspace on subsequent logins.' :
              'Review and create.'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {step === 'workspace' && (
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="ws">Workspace name</FieldLabel>
                <Input
                  id="ws"
                  value={state.workspaceName}
                  onChange={(e) => update({ workspaceName: e.target.value })}
                  placeholder="Acme Research"
                  autoFocus
                />
                <FieldDescription>
                  Visible in the dashboard and audit trail.
                </FieldDescription>
              </Field>
            </FieldGroup>
          )}

          {step === 'admin' && (
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="email">Admin email</FieldLabel>
                <Input
                  id="email"
                  type="email"
                  value={state.adminEmail}
                  onChange={(e) => update({ adminEmail: e.target.value })}
                  placeholder="you@company.com"
                  autoFocus
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="pw">Password</FieldLabel>
                <Input
                  id="pw"
                  type="password"
                  value={ state.adminPassword }
                  onChange={(e) => update({ adminPassword: e.target.value })}
                  placeholder="at least 8 characters"
                />
              </Field>
            </FieldGroup>
          )}

          {step === 'llm' && (
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="prov">Provider</FieldLabel>
                <Select
                  value={state.llmProvider}
                  onValueChange={(v) => {
                    const p = v as OnboardingState['llmProvider'];
                    update({
                      llmProvider: p,
                      llmModel: providerDefaultModel(p),
                      llmBaseUrl: providerDefaultBaseUrl(p),
                    });
                  }}
                >
                  <SelectTrigger id="prov">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="minimax">MiniMax</SelectItem>
                    <SelectItem value="litellm">LiteLLM (local proxy)</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="bedrock">AWS Bedrock</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field>
                <FieldLabel htmlFor="model">Model</FieldLabel>
                <Input
                  id="model"
                  value={state.llmModel}
                  onChange={(e) => update({ llmModel: e.target.value })}
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="apikey">API key</FieldLabel>
                <Input
                  id="apikey"
                  type="password"
                  value={state.llmApiKey}
                  onChange={(e) => update({ llmApiKey: e.target.value })}
                  placeholder="sk-..."
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="baseurl">Base URL (optional)</FieldLabel>
                <Input
                  id="baseurl"
                  value={state.llmBaseUrl}
                  onChange={(e) => update({ llmBaseUrl: e.target.value })}
                  placeholder={providerDefaultBaseUrl(state.llmProvider)}
                />
                <FieldDescription>
                  Stored in workspace_settings, never logged.
                </FieldDescription>
              </Field>
            </FieldGroup>
          )}

          {step === 'passphrase' && (
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="pp">Workspace passphrase</FieldLabel>
                <Input
                  id="pp"
                  type="password"
                  value={state.passphrase}
                  onChange={(e) => update({ passphrase: e.target.value })}
                  placeholder="at least 8 characters"
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="pp2">Confirm passphrase</FieldLabel>
                <Input
                  id="pp2"
                  type="password"
                  value={state.passphraseConfirm}
                  onChange={(e) => update({ passphraseConfirm: e.target.value })}
                />
                <FieldDescription>
                  Lose it and the encrypted keys are lost too — keep a copy
                  in your password manager.
                </FieldDescription>
              </Field>
            </FieldGroup>
          )}

          {step === 'confirm' && (
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between border-b border-border py-2">
                <dt className="text-muted-foreground">Workspace</dt>
                <dd className="font-mono text-xs">{state.workspaceName}</dd>
              </div>
              <div className="flex justify-between border-b border-border py-2">
                <dt className="text-muted-foreground">Admin</dt>
                <dd className="font-mono text-xs">{state.adminEmail}</dd>
              </div>
              <div className="flex justify-between border-b border-border py-2">
                <dt className="text-muted-foreground">LLM</dt>
                <dd className="font-mono text-xs">{state.llmProvider} / {state.llmModel}</dd>
              </div>
              <div className="flex justify-between border-b border-border py-2">
                <dt className="text-muted-foreground">Passphrase</dt>
                <dd className="font-mono text-xs">••••••••</dd>
              </div>
            </dl>
          )}

          {error && (
            <p className="text-sm text-destructive" role="alert">
              {error}
            </p>
          )}

          <div className="flex justify-between pt-2">
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
          <p className="text-center text-xs text-muted-foreground">
            Already have a workspace?{' '}
            <Link href="/sign-in" className="underline">
              Sign in
            </Link>
            .
          </p>
        </CardContent>
      </Card>
    </div>
  );
}