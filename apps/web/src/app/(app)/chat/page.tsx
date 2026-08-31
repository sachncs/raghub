"use client";

import * as React from "react";
import { motion, AnimatePresence } from "motion/react";
import { toast } from "sonner";
import {
  ArrowUp,
  CircleNotch,
  ScanSearch,
  ShieldCheck,
  Sparkles,
} from "@/lib/icons";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Kbd } from "@/components/ui/kbd";
import { Progress } from "@/components/ui/progress";
import { useRevexStream } from "@/lib/hooks/use-revex-stream";
import { cn } from "@/lib/utils";

interface ChatPageProps {
  readonly seedQuestion?: string;
}

interface ChatMessage {
  readonly id: number;
  readonly role: "user" | "assistant";
  readonly text: string;
}

const STARTERS: readonly { label: string; query: string }[] = [
  { label: "Find recent mentions of vendor X", query: "Find recent mentions of vendor X." },
  { label: "Summarise the Q3 contract", query: "Summarise the Q3 contract." },
  { label: "Who owns the pricing memo?", query: "Who owns the pricing memo?" },
  { label: "What changed in security policy?", query: "What changed in security policy?" },
];

export default function ChatPage({ seedQuestion }: ChatPageProps) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [input, setInput] = React.useState("");
  const [sessionId] = React.useState(() =>
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `s-${Date.now().toString(36)}`
  );
  const nextId = React.useRef(1);
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const composerRef = React.useRef<HTMLTextAreaElement>(null);
  const stream = useRevexStream();

  React.useEffect(() => {
    if (!document.cookie.includes("revex_session=")) {
      window.location.href = "/sign-in";
    }
  }, []);

  React.useEffect(() => {
    if (seedQuestion) setInput(seedQuestion);
  }, [seedQuestion]);

  React.useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  React.useEffect(() => {
    if (stream.error) toast.error(stream.error);
  }, [stream.error]);

  const assistantMessageIdRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const id = assistantMessageIdRef.current;
    if (id === null || !stream.text) return;
    setMessages((m) =>
      m.map((msg) => (msg.id === id ? { ...msg, text: String(stream.text) } : msg))
    );
  }, [stream.text]);

  const submit = async (override?: string): Promise<void> => {
    const question = (override ?? input).trim();
    if (!question || stream.streaming) return;
    const userMsg: ChatMessage = {
      id: nextId.current++,
      role: "user",
      text: question,
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    const assistantId = nextId.current++;
    setMessages((m) => [...m, { id: assistantId, role: "assistant", text: "" }]);

    assistantMessageIdRef.current = assistantId;
    const text = await stream.start({ question, sessionId });
    setMessages((m) =>
      m.map((msg) =>
        msg.id === assistantId ? { ...msg, text: text || stream.text } : msg
      )
    );

    const turnId = `${sessionId}:${assistantId}`;
    void fetch("/api/proxy", {
      method: "POST",
      headers: { "content-type": "application/json", "x-revex-path": "/v1/feedback" },
      body: JSON.stringify({ turnId, rating: "neutral" }),
    }).catch(() => undefined);
    try {
      window.localStorage.setItem(`revex:lastTurnId:${assistantId}`, turnId);
    } catch {
      /* ignore */
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submit();
    }
  };

  const rate = async (messageId: number, rating: "up" | "down"): Promise<void> => {
    const turnId =
      (typeof window !== "undefined"
        ? window.localStorage.getItem(`revex:lastTurnId:${messageId}`)
        : null) ?? `${sessionId}:${messageId}`;
    try {
      await fetch("/api/proxy", {
        method: "POST",
        headers: { "content-type": "application/json", "x-revex-path": "/v1/feedback" },
        body: JSON.stringify({ turnId, rating }),
      });
      toast.success(`Marked as ${rating === "up" ? "helpful" : "needs work"}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "feedback failed");
    }
  };

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] flex-col">
      <ContextStrip sessionId={sessionId} streaming={stream.streaming} />
      {stream.streaming && (
        <div className="sticky top-14 z-10 -mt-px">
          <Progress value={70} className="h-0.5 rounded-none" />
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 py-10">
          {messages.length === 0 ? (
            <EmptyChat onPrompt={(q) => void submit(q)} disabled={stream.streaming} />
          ) : (
            <ol className="flex flex-col gap-6">
              {messages.map((m) => (
                <li key={m.id}>
                  <MessageBubble message={m} onRate={rate} />
                </li>
              ))}
              <div ref={scrollRef} />
            </ol>
          )}
        </div>
      </div>

      <div className="sticky bottom-0 z-10 border-t bg-background/85 backdrop-blur-md">
        <div className="mx-auto w-full max-w-3xl px-4 py-4">
          <div
            className={cn(
              "flex items-end gap-2 rounded-2xl border border-border bg-card/80 p-2 shadow-sm transition-shadow",
              "focus-within:border-primary/40 focus-within:shadow-md focus-within:shadow-primary/10"
            )}
          >
            <textarea
              ref={composerRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask anything about your workspace…"
              rows={1}
              className="min-h-9 max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm leading-relaxed outline-none placeholder:text-muted-foreground"
            />
            {stream.streaming ? (
              <Button
                size="icon"
                variant="destructive"
                onClick={stream.stop}
                aria-label="Stop generating"
              >
                <CircleNotch className="size-4 animate-spin" />
              </Button>
            ) : (
              <Button
                size="icon"
                onClick={() => void submit()}
                disabled={!input.trim()}
                aria-label="Send"
                className="size-9 rounded-xl"
              >
                <ArrowUp className="size-4" />
              </Button>
            )}
          </div>
          <div className="mt-2 flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
            <span>
              Press <Kbd>Enter</Kbd> to send · <Kbd>Shift</Kbd>+<Kbd>Enter</Kbd> for newline
            </span>
            <span className="inline-flex items-center gap-1.5">
              <ShieldCheck className="size-3" />
              Policy-scoped
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ContextStrip({
  sessionId,
  streaming,
}: {
  readonly sessionId: string;
  readonly streaming: boolean;
}) {
  return (
    <div className="border-b bg-background/60 px-4 py-2.5 backdrop-blur-md">
      <div className="mx-auto flex max-w-3xl items-center justify-between text-xs">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Sparkles className="size-3.5 text-primary" />
          <span className="font-medium text-foreground">Hybrid retrieval</span>
          <span className="text-muted-foreground/50">·</span>
          <span className="font-mono">{sessionId.slice(0, 8)}</span>
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Badge variant="outline" className="h-5 gap-1 px-1.5 py-0 text-[10px] uppercase tracking-wider">
            <span
              className={cn(
                "size-1.5 rounded-full",
                streaming ? "bg-amber-500 animate-pulse" : "bg-emerald-500"
              )}
              aria-hidden
            />
            {streaming ? "Streaming" : "Ready"}
          </Badge>
        </div>
      </div>
    </div>
  );
}

function EmptyChat({
  onPrompt,
  disabled,
}: {
  onPrompt: (q: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex min-h-[40vh] flex-col items-center justify-center py-12 text-center">
      <div className="mb-6 inline-flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500/20 to-amber-500/20 ring-1 ring-border/40">
        <ScanSearch className="size-7 text-primary" />
      </div>
      <h1 className="text-h2 mb-2 text-balance text-foreground">
        What can I help you retrieve?
      </h1>
      <p className="mx-auto max-w-md text-pretty text-sm text-muted-foreground">
        Revex fans out across vector, keyword, graph, memory, and web — then
        fuses the results into one policy-aware answer.
      </p>
      <div className="mt-8 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {STARTERS.map((s) => (
          <button
            key={s.label}
            type="button"
            disabled={disabled}
            onClick={() => onPrompt(s.query)}
            className="group flex items-start gap-3 rounded-xl border border-border/60 bg-card/50 p-3 text-left transition-all hover:-translate-y-0.5 hover:border-border hover:bg-card disabled:opacity-50"
          >
            <span className="mt-1 inline-flex size-5 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
              <Sparkles className="size-3" />
            </span>
            <span className="text-sm font-medium text-foreground">
              {s.label}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  onRate,
}: {
  message: ChatMessage;
  onRate: (id: number, rating: "up" | "down") => void;
}) {
  const isUser = message.role === "user";
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
        className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
      >
        {isUser ? (
          <Card className="max-w-[85%] border-primary/30 bg-primary px-4 py-3 text-primary-foreground shadow-sm">
            <div className="whitespace-pre-wrap text-sm leading-relaxed">
              {message.text}
            </div>
          </Card>
        ) : (
          <Card className="max-w-[85%] border-border/60 bg-card/70 px-4 py-3 shadow-sm">
            {message.text.length === 0 ? (
              <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                <span className="size-1.5 animate-pulse rounded-full bg-current" />
                <span className="size-1.5 animate-pulse rounded-full bg-current [animation-delay:150ms]" />
                <span className="size-1.5 animate-pulse rounded-full bg-current [animation-delay:300ms]" />
              </span>
            ) : (
              <>
                <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                  {message.text}
                </div>
                <div className="mt-3 flex items-center gap-1 border-t border-border/40 pt-2 text-[11px] text-muted-foreground">
                  <button
                    type="button"
                    onClick={() => onRate(message.id, "up")}
                    className="inline-flex items-center gap-1 rounded-md px-2 py-1 transition-colors hover:bg-muted hover:text-foreground"
                  >
                    Helpful
                  </button>
                  <span className="text-muted-foreground/30">·</span>
                  <button
                    type="button"
                    onClick={() => onRate(message.id, "down")}
                    className="inline-flex items-center gap-1 rounded-md px-2 py-1 transition-colors hover:bg-muted hover:text-foreground"
                  >
                    Needs work
                  </button>
                </div>
              </>
            )}
          </Card>
        )}
      </motion.div>
    </AnimatePresence>
  );
}