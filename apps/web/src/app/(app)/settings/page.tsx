"use client";

import * as React from "react";
import { toast } from "sonner";
import { Key, Lightning, Shield, User as UserIcon } from "@/lib/icons";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";

type Provider = 'openai' | 'anthropic' | 'bedrock' | 'litellm' | 'minimax';

interface LlmSettings {
  provider: Provider;
  model: string;
  apiKey?: string;
  baseUrl?: string;
  temperature?: number;
}

const PROVIDER_MODELS: Record<Provider, string> = {
  openai: 'gpt-4.1',
  anthropic: 'claude-3-5-sonnet-latest',
  bedrock: 'anthropic.claude-3-5-sonnet-20241022-v2',
  litellm: 'gpt-4.1',
  minimax: 'MiniMax-Text-01',
};

const proxy = async (path: string, init: RequestInit = {}): Promise<Response> =>
  fetch("/api/proxy", {
    ...init,
    headers: { ...init.headers, "x-revex-path": path },
  });

export default function SettingsPage() {
  const [llm, setLlm] = React.useState<LlmSettings | null>(null);
  const [apiKey, setApiKey] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState("llm");

  const refresh = React.useCallback(async (): Promise<void> => {
    const res = await proxy("/v1/settings/llm");
    if (!res.ok) {
      const body = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
      setError(body?.error?.message ?? "Failed to load");
      return;
    }
    const body = (await res.json()) as { llm: LlmSettings | null };
    setLlm(body.llm);
    setApiKey("");
  }, []);

  React.useEffect(() => {
    if (!document.cookie.includes("revex_session=")) {
      window.location.href = "/sign-in";
      return;
    }
    void refresh();
  }, [refresh]);

  const save = async (): Promise<void> => {
    if (!llm) return;
    setSaving(true);
    setError(null);
    const body = {
      provider: llm.provider,
      model: llm.model,
      ...(apiKey ? { apiKey } : {}),
      ...(llm.baseUrl ? { baseUrl: llm.baseUrl } : {}),
      ...(llm.temperature !== undefined ? { temperature: llm.temperature } : {}),
    };
    const res = await proxy("/v1/settings/llm", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    setSaving(false);
    if (!res.ok) {
      const errBody = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
      const message = errBody?.error?.message ?? "Save failed";
      setError(message);
      toast.error(message);
      return;
    }
    toast.success("Saved");
    setApiKey("");
    void refresh();
  };

  if (!llm) {
    return (
      <div className="container max-w-3xl px-6 py-8">
        <h1 className="mb-6 text-2xl font-semibold tracking-tight">Settings</h1>
        <div className="space-y-3">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-48 w-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="container max-w-3xl px-6 py-8">
      <h1 className="mb-6 text-2xl font-semibold tracking-tight">Settings</h1>
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-6">
          <TabsTrigger value="llm">
            <Lightning className="size-4" />
            LLM
          </TabsTrigger>
          <TabsTrigger value="account">
            <UserIcon className="size-4" />
            Account
          </TabsTrigger>
          <TabsTrigger value="workspace">
            <Key className="size-4" />
            Workspace
          </TabsTrigger>
          <TabsTrigger value="security">
            <Shield className="size-4" />
            Security
          </TabsTrigger>
        </TabsList>

        <TabsContent value="llm">
          <Card>
            <CardHeader>
              <CardTitle>LLM provider</CardTitle>
              <CardDescription>
                Revex stores API keys encrypted with the workspace passphrase.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="prov">Provider</Label>
                <Select
                  value={llm.provider}
                  onValueChange={(v) =>
                    setLlm({
                      ...llm,
                      provider: v as Provider,
                      model: PROVIDER_MODELS[v as Provider],
                    })
                  }
                >
                  <SelectTrigger id="prov">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="bedrock">AWS Bedrock</SelectItem>
                    <SelectItem value="litellm">LiteLLM</SelectItem>
                    <SelectItem value="minimax">minimax</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="model">Model</Label>
                <Input
                  id="model"
                  value={llm.model}
                  onChange={(e) => setLlm({ ...llm, model: e.target.value })}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="apikey">API key</Label>
                <Input
                  id="apikey"
                  type="password"
                  placeholder={llm.apiKey ?? "leave blank to keep current"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Encrypted with the workspace passphrase before being stored.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="baseurl">Base URL (optional)</Label>
                <Input
                  id="baseurl"
                  value={llm.baseUrl ?? ""}
                  onChange={(e) => {
                    const next = e.target.value;
                    const updated: LlmSettings = { ...llm };
                    if (next) updated.baseUrl = next;
                    else delete (updated as { baseUrl?: string }).baseUrl;
                    setLlm(updated);
                  }}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="temp">Temperature</Label>
                <Input
                  id="temp"
                  type="number"
                  step={0.1}
                  min={0}
                  max={2}
                  value={llm.temperature ?? 0}
                  onChange={(e) => setLlm({ ...llm, temperature: Number(e.target.value) })}
                />
              </div>
              {error && (
                <p className="text-sm text-destructive" role="alert">
                  {error}
                </p>
              )}
              <Separator />
              <div className="flex justify-end">
                <Button onClick={save} disabled={saving}>
                  {saving ? "Saving…" : "Save changes"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="account">
          <Card>
            <CardHeader>
              <CardTitle>Account</CardTitle>
              <CardDescription>
                Your account details. Editing isn&apos;t wired in this release.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Display name</span>
                <span className="font-mono text-xs">—</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Email</span>
                <span className="font-mono text-xs">—</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="workspace">
          <Card>
            <CardHeader>
              <CardTitle>Workspace</CardTitle>
              <CardDescription>
                Rename, export, or delete this workspace.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="ws-name">Workspace name</Label>
                <Input id="ws-name" placeholder="Acme Research" disabled />
              </div>
              <Button variant="outline" disabled>
                Export workspace…
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security">
          <Card>
            <CardHeader>
              <CardTitle>Security</CardTitle>
              <CardDescription>
                Encryption, sessions, and audit trail.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Encryption</span>
                <span className="font-mono text-xs">AES-256-GCM (passphrase-derived)</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Session TTL</span>
                <span className="font-mono text-xs">24h</span>
              </div>
              <Separator />
              <Button variant="outline" disabled>
                Rotate passphrase…
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}