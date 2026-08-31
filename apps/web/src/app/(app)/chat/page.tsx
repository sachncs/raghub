"use client";

import * as React from "react";
import { motion } from "motion/react";
import {
  ArrowUp,
  MagnifyingGlass,
  Sparkles,
  Stop,
  Lightning,
} from "@/lib/icons";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { EmptyState } from "@/components/state";
import { useRevexStream, type StreamEvent } from "@/lib/hooks/use-revex-stream";
import { cn } from "@/lib/utils";

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  events: readonly StreamEvent[];
  streaming?: boolean;
}

interface ChatPageProps {
  readonly seedQuestion?: string;
}

const STARTERS: readonly { label: string; query: string }[] = [
  { label: "Find recent mentions of vendor X", query: "Find recent mentions of vendor X across the workspace." },
  { label: "Summarise the Q3 contract", query: "Summarise the Q3 contract and list the key obligations." },
  { label: "Who owns the pricing memo?", query: "Who owns the pricing memo? Show the latest version." },
  { label: "What changed in the security policy?", query: "What changed in the security policy this month?" },
];

export default function ChatPage({ seedQuestion }: ChatPageProps) {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const [sessionId] = React.useState(() =>
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `s_${Date.now().toString(36)}`
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
    if (stream.error) {
      toast.error(stream.error);
    }
  }, [stream.error]);

  const send = async (override?: string): Promise<void> => {
    const question = (override ?? input).trim();
    if (!question || stream.streaming) return;
    const userMsg: Message = {
      id: nextId.current++,
      role: "user",
      text: question,
      events: [],
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    const assistantId = nextId.current++;
    setMessages((m) => [
      ...m,
      { id: assistantId, role: "assistant", text: "", events: [], streaming: true },
    ]);

    await stream.start({ question, sessionId });

    setMessages((m) =>
      m.map((msg) =>
        msg.id === assistantId
          ? {
              ...msg,
              text: stream.text,
              events: stream.events,
              streaming: false,
            }
          : msg
      )
    );
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] flex-col">
      <ChatToolbar sessionId={sessionId} hasEvents={messages.some((m) => m.events.length > 0)} />

      <ScrollArea className="flex-1">
        <div className="mx-auto w-full max-w-3xl px-4 py-8">
          {messages.length === 0 ? (
            <EmptyChat onPrompt={(q) => void send(q)} disabled={stream.streaming} />
          ) : (
            <ol className="flex flex-col gap-8">
              {messages.map((m) => (
                <li key={m.id}>
                  <MessageBubble message={m} />
                </li>
              ))}
              <div ref={scrollRef} />
            </ol>
          )}
        </div>
      </ScrollArea>

      {stream.streaming && (
        <div className="sticky top-14 z-10 -mt-px">
          <Progress value={70} className="h-0.5 rounded-none" />
        </div>
      )}

      <div className="sticky bottom-0 z-10 border-t bg-background/80 backdrop-blur-md">
        <div className="mx-auto w-full max-w-3xl px-4 py-4">
          <div
            className={cn(
              "flex items-end gap-2 rounded-2xl border border-border bg-card/80 p-2 shadow-sm transition-shadow",
              "focus-within:border-primary/40 focus-within:shadow-md"
            )}
          >
            <textarea
              ref={composerRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={`Ask Revex…`}
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
                <Stop className="size-4" />
              </Button>
            ) : (
              <Button
                size="icon"
                onClick={() => void send()}
                disabled={!input.trim()}
                aria-label="Send"
              >
                <ArrowUp className="size-4" />
              </Button>
            )}
          </div>
          <p className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
            <span>
              Press <Kbd>Enter</Kbd> to send · <Kbd>Shift</Kbd>+<Kbd>Enter</Kbd> for newline
            </span>
            <span className="font-mono">session {sessionId.slice(0, 8)}…</span>
          </p>
        </div>
      </div>
    </div>
  );
}

function ChatToolbar({
  sessionId,
  hasEvents,
}: {
  sessionId: string;
  hasEvents: boolean;
}) {
  return (
    <div className="sticky top-14 z-10 -mt-px flex h-12 items-center gap-2 border-b bg-background/80 px-4 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-3xl items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Sparkles className="size-4" />
          <span className="font-medium">Hybrid retrieval</span>
          <span className="text-muted-foreground/50">·</span>
          <span className="font-mono text-xs">{sessionId.slice(0, 8)}…</span>
        </div>
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="ghost" size="sm" disabled={!hasEvents}>
              <MagnifyingGlass className="size-4" />
              Sub-agent traces
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="w-full sm:max-w-md">
            <SheetHeader>
              <SheetTitle>Sub-agent traces</SheetTitle>
              <SheetDescription>
                Every tool call and retrieval in this conversation.
              </SheetDescription>
            </SheetHeader>
            <TraceList />
          </SheetContent>
        </Sheet>
      </div>
    </div>
  );
}

function TraceList() {
  const lastEvents = (() => {
    if (typeof window === "undefined") return [];
    const root = document.querySelector('[data-chat-traces]');
    if (!root) return [];
    try {
      return JSON.parse(root.getAttribute("data-chat-traces") ?? "[]");
    } catch {
      return [];
    }
  })();
  if (lastEvents.length === 0) {
    return (
      <EmptyState
        icon={Lightning}
        title="No traces yet"
        description="Send a message to see the sub-agent activity for this conversation."
        className="mt-6"
      />
    );
  }
  return (
    <ol className="mt-6 space-y-2">
      {(lastEvents as StreamEvent[]).map((e) => (
        <li
          key={e.step}
          className="rounded-md border border-border/60 bg-card p-2 font-mono text-xs"
        >
          <div className="flex items-center justify-between text-muted-foreground">
            <span>step {e.step}</span>
            <span className="rounded bg-muted px-1.5 py-0.5">{e.kind}</span>
          </div>
          <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-[11px] text-foreground">
            {JSON.stringify(e.payload, null, 2)}
          </pre>
        </li>
      ))}
    </ol>
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
    <div className="flex min-h-[50vh] flex-col items-center justify-center py-12 text-center">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="mx-auto mb-6 inline-flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500/20 to-amber-500/20">
          <Sparkles className="size-7" />
        </div>
        <h1 className="mb-2 text-3xl font-semibold tracking-tight">
          What can I help you retrieve?
        </h1>
        <p className="mx-auto max-w-md text-pretty text-muted-foreground">
          Revex fans out across vector, keyword, graph, memory, and web —
          then fuses the results into one ranked answer.
        </p>
      </motion.div>
      <div className="mt-8 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {STARTERS.map((s) => (
          <button
            key={s.label}
            type="button"
            disabled={disabled}
            onClick={() => onPrompt(s.query)}
            className="group flex items-start gap-3 rounded-xl border border-border/60 bg-card/50 p-3 text-left transition-all hover:-translate-y-0.5 hover:border-border hover:bg-card disabled:opacity-50"
          >
            <span className="mt-0.5 size-1.5 rounded-full bg-primary/70 transition-all group-hover:bg-primary" />
            <span className="text-sm font-medium">{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-card text-foreground border border-border/60"
        )}
      >
        {message.streaming && !message.text ? (
          <span className="inline-flex items-center gap-1.5 text-muted-foreground">
            <span className="size-1.5 animate-pulse rounded-full bg-current" />
            <span className="size-1.5 animate-pulse rounded-full bg-current [animation-delay:150ms]" />
            <span className="size-1.5 animate-pulse rounded-full bg-current [animation-delay:300ms]" />
          </span>
        ) : (
          <div className="whitespace-pre-wrap">{message.text || (message.streaming ? "…" : "")}</div>
        )}
      </div>
    </motion.div>
  );
}