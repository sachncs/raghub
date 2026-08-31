import * as React from "react";
import Link from "next/link";

import {
  Brain,
  CircuitBoard,
  Command,
  FileText,
  GitBranch,
  Globe,
  KeyRound,
  MessagesSquare,
  Network,
  Radar,
  ScanSearch,
  ShieldCheck,
  Users,
} from "@/lib/icons";
import { LogoMark } from "@/components/marketing/logo-mark";
import { Button } from "@/components/ui/button";
import { CommandPalette, useCommandPalette } from "@/components/command-palette";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
} from "@/components/ui/navigation-menu";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { Moon, SunDim } from "@/lib/icons";
import { useTheme } from "next-themes";

interface ProductItem {
  readonly title: string;
  readonly description: string;
  readonly href: string;
  readonly icon: React.ComponentType<{ className?: string }>;
}

const PRODUCT_ITEMS: readonly ProductItem[] = [
  {
    title: "Hybrid retrieval",
    description: "Vector, keyword, graph, memory, web — fused behind one engine.",
    href: "/#capabilities",
    icon: Network,
  },
  {
    title: "Reranking",
    description: "Cohere, ColBERT MaxSim, LLM-as-judge, custom scorers.",
    href: "/#capabilities",
    icon: Radar,
  },
  {
    title: "Generation",
    description: "Policy-aware answer assembly with citations.",
    href: "/#capabilities",
    icon: Brain,
  },
  {
    title: "Governance",
    description: "Document-level ACLs, per-user strategy, audit trail.",
    href: "/#policy",
    icon: ShieldCheck,
  },
  {
    title: "Indexing",
    description: "Connectors for files, APIs, databases, webhooks.",
    href: "/#platform",
    icon: ScanSearch,
  },
  {
    title: "Workspace",
    description: "Sealed SQLite + passphrase. Self-host or cloud.",
    href: "/#platform",
    icon: KeyRound,
  },
];

const SOLUTIONS_ITEMS: readonly ProductItem[] = [
  {
    title: "Knowledge base search",
    description: "Confluence, Notion, Drive — grounded answers.",
    href: "/#capabilities",
    icon: FileText,
  },
  {
    title: "Internal copilots",
    description: "Slack and Teams bots grounded in your docs.",
    href: "/#capabilities",
    icon: MessagesSquare,
  },
  {
    title: "Customer support",
    description: "Zendesk answers and ticket escalation.",
    href: "/#capabilities",
    icon: Users,
  },
  {
    title: "Compliance & legal",
    description: "Contract review with full audit trail.",
    href: "/#policy",
    icon: ShieldCheck,
  },
  {
    title: "Research",
    description: "Papers, patents, internal experiments.",
    href: "/#capabilities",
    icon: Globe,
  },
  {
    title: "Operational intel",
    description: "Runbooks, postmortems, on-call context.",
    href: "/#platform",
    icon: CircuitBoard,
  },
];

export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-border/40 bg-background/70 backdrop-blur-md">
      <div className="container flex h-16 items-center justify-between gap-6">
        <Link
          href="/"
          aria-label="Revex home"
          className="inline-flex items-center gap-2.5"
        >
          <LogoMark size="sm" />
          <span className="text-base font-semibold italic tracking-tight">
            re
            <span className="not-italic text-primary">›</span>
            ex
          </span>
        </Link>
        <DesktopNav />
        <HeaderActions />
      </div>
    </header>
  );
}

function DesktopNav() {
  return (
    <NavigationMenu className="hidden md:flex" aria-label="Primary">
      <NavigationMenuList>
        <NavigationMenuItem>
          <NavigationMenuTrigger>Product</NavigationMenuTrigger>
          <NavigationMenuContent>
            <ul className="grid w-[640px] gap-1 p-3 md:grid-cols-2">
              {PRODUCT_ITEMS.map((item) => (
                <ListItem key={item.title} item={item} />
              ))}
            </ul>
          </NavigationMenuContent>
        </NavigationMenuItem>
        <NavigationMenuItem>
          <NavigationMenuTrigger>Solutions</NavigationMenuTrigger>
          <NavigationMenuContent>
            <ul className="grid w-[640px] gap-1 p-3 md:grid-cols-2">
              {SOLUTIONS_ITEMS.map((item) => (
                <ListItem key={item.title} item={item} />
              ))}
            </ul>
          </NavigationMenuContent>
        </NavigationMenuItem>
        <NavigationMenuItem>
          <NavigationMenuLink asChild>
            <Link
              href="/#policy"
              className="inline-flex h-9 items-center rounded-md px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            >
              Security
            </Link>
          </NavigationMenuLink>
        </NavigationMenuItem>
        <NavigationMenuItem>
          <NavigationMenuLink asChild>
            <Link
              href="/docs"
              className="inline-flex h-9 items-center rounded-md px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            >
              Docs
            </Link>
          </NavigationMenuLink>
        </NavigationMenuItem>
        <NavigationMenuItem>
          <NavigationMenuLink asChild>
            <Link
              href="/pricing"
              className="inline-flex h-9 items-center rounded-md px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            >
              Pricing
            </Link>
          </NavigationMenuLink>
        </NavigationMenuItem>
      </NavigationMenuList>
    </NavigationMenu>
  );
}

function ListItem({ item }: { readonly item: ProductItem }) {
  return (
    <li>
      <NavigationMenuLink asChild>
        <Link
          href={item.href}
          className="group flex items-start gap-3 rounded-xl p-3 transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        >
          <span className="mt-0.5 inline-flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-foreground/80 transition-colors group-hover:bg-primary/10 group-hover:text-primary">
            <item.icon className="size-4" />
          </span>
          <span className="flex flex-col gap-0.5">
            <span className="text-sm font-medium text-foreground">
              {item.title}
            </span>
            <span className="text-xs leading-relaxed text-muted-foreground">
              {item.description}
            </span>
          </span>
        </Link>
      </NavigationMenuLink>
    </li>
  );
}

function HeaderActions() {
  const { setTheme, resolvedTheme } = useTheme();
  const palette = useCommandPalette();

  return (
    <div className="flex items-center gap-1.5">
      <button
        type="button"
        onClick={palette.toggle}
        aria-label="Open command palette"
        className="hidden h-9 items-center gap-2 rounded-lg border border-border/60 bg-background/40 px-3 text-xs text-muted-foreground transition-colors hover:border-border hover:bg-background/80 hover:text-foreground md:inline-flex"
      >
        <Command className="size-3.5" />
        <span>Quick search</span>
        <KbdGroup className="ml-2">
          <Kbd>⌘</Kbd>
          <Kbd>K</Kbd>
        </KbdGroup>
      </button>
      <button
        type="button"
        onClick={palette.toggle}
        aria-label="Open command palette"
        className="inline-flex size-9 items-center justify-center rounded-lg border border-border/60 bg-background/40 text-muted-foreground transition-colors hover:border-border hover:bg-background/80 hover:text-foreground md:hidden"
      >
        <Command className="size-4" />
      </button>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
        aria-label="Toggle theme"
      >
        <SunDim className="size-4 dark:hidden" />
        <Moon className="hidden size-4 dark:block" />
      </Button>
      <Button asChild variant="ghost" size="sm" className="hidden sm:inline-flex">
        <Link href="/sign-in">Sign in</Link>
      </Button>
      <Button asChild size="sm" className="hidden md:inline-flex">
        <Link href="/onboarding">Create workspace</Link>
      </Button>
      <Link
        href="/onboarding"
        aria-label="Create workspace"
        className="inline-flex size-9 items-center justify-center rounded-md bg-primary text-primary-foreground md:hidden"
      >
        <GitBranch className="size-4" />
      </Link>
      <CommandPalette open={palette.open} setOpen={palette.setOpen} />
    </div>
  );
}