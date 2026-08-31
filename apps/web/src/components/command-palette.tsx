"use client";

import * as React from "react";
import { useRouter } from "next/navigation";

import {
  ChatCircle,
  FileText,
  Gear,
  MagnifyingGlass,
  Users,
} from "@/lib/icons";

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";

interface CommandPaletteState {
  readonly open: boolean;
  readonly setOpen: (open: boolean) => void;
  readonly toggle: () => void;
}

export function useCommandPalette(): CommandPaletteState {
  const [open, setOpen] = React.useState(false);
  const toggle = React.useCallback(() => setOpen((v) => !v), []);

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [toggle]);

  return { open, setOpen, toggle };
}

interface CommandPaletteProps {
  readonly open: boolean;
  readonly setOpen: (open: boolean) => void;
}

interface Command {
  readonly id: string;
  readonly label: string;
  readonly href: string;
  readonly icon: React.ComponentType<{ className?: string }>;
  readonly group: string;
  readonly shortcut?: string;
}

const COMMANDS: readonly Command[] = [
  { id: "chat", label: "Go to Chat", href: "/chat", icon: ChatCircle, group: "Navigate", shortcut: "G C" },
  { id: "documents", label: "Go to Documents", href: "/documents", icon: FileText, group: "Navigate", shortcut: "G D" },
  { id: "members", label: "Go to Members", href: "/members", icon: Users, group: "Navigate", shortcut: "G M" },
  { id: "settings", label: "Go to Settings", href: "/settings", icon: Gear, group: "Navigate", shortcut: "G S" },
  { id: "search", label: "Search workspace", href: "/chat", icon: MagnifyingGlass, group: "Actions", shortcut: "/" },
];

export function CommandPalette({ open, setOpen }: CommandPaletteProps) {
  const router = useRouter();

  const handleSelect = React.useCallback(
    (href: string) => {
      setOpen(false);
      router.push(href);
    },
    [router, setOpen]
  );

  const groups = React.useMemo(() => {
    const byGroup = new Map<string, Command[]>();
    for (const cmd of COMMANDS) {
      const list = byGroup.get(cmd.group) ?? [];
      list.push(cmd);
      byGroup.set(cmd.group, list);
    }
    return Array.from(byGroup.entries());
  }, []);

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Type a command or search…" />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        {groups.map(([group, items]) => (
          <CommandGroup key={group} heading={group}>
            {items.map((cmd) => (
              <CommandItem
                key={cmd.id}
                value={cmd.label}
                onSelect={() => handleSelect(cmd.href)}
              >
                <cmd.icon className="size-4" />
                <span>{cmd.label}</span>
                {cmd.shortcut && <CommandShortcut>{cmd.shortcut}</CommandShortcut>}
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </CommandDialog>
  );
}