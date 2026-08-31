"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  ChatCircle,
  FileText,
  Gear,
  Key,
  LogOut,
  MagnifyingGlass,
  Moon,
  PanelLeft,
  SunDim,
  Users,
} from "@/lib/icons";
import { useTheme } from "next-themes";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Wordmark } from "@/components/wordmark";
import { UserAvatar } from "@/components/user-avatar";
import { CommandPalette, useCommandPalette } from "@/components/command-palette";

interface NavLink {
  readonly href: string;
  readonly label: string;
  readonly icon: React.ComponentType<{ className?: string }>;
}

const NAV_PRIMARY: readonly NavLink[] = [
  { href: "/chat", label: "Chat", icon: ChatCircle },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/members", label: "Members", icon: Users },
];

const NAV_SECONDARY: readonly NavLink[] = [
  { href: "/settings", label: "Settings", icon: Gear },
];

interface AppShellProps {
  readonly children: React.ReactNode;
  readonly user?: {
    readonly email?: string | null;
    readonly displayName?: string | null;
  };
}

export function AppShell({ children, user }: AppShellProps) {
  const pathname = usePathname() ?? "";
  const router = useRouter();
  const palette = useCommandPalette();

  const signOut = React.useCallback(async () => {
    try {
      await fetch("/api/proxy", {
        method: "POST",
        headers: { "x-revex-path": "/v1/auth/logout" },
      });
    } finally {
      document.cookie =
        "revex_session=; path=/; max-age=0; samesite=lax";
      document.cookie =
        "revex_workspace_key=; path=/; max-age=0; samesite=lax";
      router.push("/sign-in");
    }
  }, [router]);

  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader>
          <Link href="/" className="flex items-center">
            <Wordmark size="md" />
          </Link>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Workspace</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV_PRIMARY.map((link) => (
                  <SidebarMenuItem key={link.href}>
                    <SidebarMenuButton
                      asChild
                      isActive={
                        pathname === link.href ||
                        pathname.startsWith(`${link.href}/`)
                      }
                      tooltip={link.label}
                    >
                      <Link href={link.href}>
                        <link.icon className="size-4" />
                        <span>{link.label}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
          <SidebarGroup className="mt-auto">
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV_SECONDARY.map((link) => (
                  <SidebarMenuItem key={link.href}>
                    <SidebarMenuButton
                      asChild
                      isActive={
                        pathname === link.href ||
                        pathname.startsWith(`${link.href}/`)
                      }
                      tooltip={link.label}
                    >
                      <Link href={link.href}>
                        <link.icon className="size-4" />
                        <span>{link.label}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
                <SidebarMenuItem>
                  <SidebarMenuButton
                    onClick={() => palette.setOpen(true)}
                    tooltip="Search (Cmd+K)"
                  >
                    <MagnifyingGlass className="size-4" />
                    <span>Search</span>
                    <Kbd className="ml-auto">⌘K</Kbd>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter>
          <UserMenu user={user} onSignOut={signOut} />
        </SidebarFooter>
        <SidebarRail />
      </Sidebar>
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-14 shrink-0 items-center gap-2 border-b bg-background/80 px-4 backdrop-blur-md">
          <SidebarTriggerInline />
          <Separator orientation="vertical" className="h-5" />
          <Breadcrumbs pathname={pathname} />
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => palette.setOpen(true)}
              className="hidden md:inline-flex"
            >
              <MagnifyingGlass className="size-4" />
              <span className="text-muted-foreground">Search</span>
              <Kbd className="ml-2">⌘K</Kbd>
            </Button>
            <ThemeToggle />
          </div>
        </header>
        <div className="flex-1 bg-background/40">{children}</div>
      </SidebarInset>
      <CommandPalette {...palette} />
    </SidebarProvider>
  );
}

function SidebarTriggerInline() {
  return (
    <button
      type="button"
      onClick={() => {
        const evt = new CustomEvent("revex:toggle-sidebar");
        window.dispatchEvent(evt);
      }}
      className="-ml-1 inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      aria-label="Toggle sidebar"
    >
      <PanelLeft className="size-4" />
    </button>
  );
}

function Breadcrumbs({ pathname }: { pathname: string }) {
  const segments = pathname.split("/").filter(Boolean);
  const last = segments[segments.length - 1] ?? "Home";
  const label = last.charAt(0).toUpperCase() + last.slice(1);
  return (
    <nav aria-label="Breadcrumb" className="text-sm">
      <ol className="flex items-center gap-1.5 text-muted-foreground">
        <li>
          <Link href="/" className="hover:text-foreground transition-colors">
            Revex
          </Link>
        </li>
        {segments.map((segment, i) => {
          const href = "/" + segments.slice(0, i + 1).join("/");
          const isLast = i === segments.length - 1;
          return (
            <li key={href} className="flex items-center gap-1.5">
              <span className="text-muted-foreground/50">/</span>
              {isLast ? (
                <span className="font-medium text-foreground">{label}</span>
              ) : (
                <Link
                  href={href}
                  className="hover:text-foreground transition-colors"
                >
                  {segment}
                </Link>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function ThemeToggle() {
  const { setTheme } = useTheme();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Toggle theme">
          <SunDim className="size-4 dark:hidden" />
          <Moon className="hidden size-4 dark:block" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Theme</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => setTheme("light")}>Light</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>Dark</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>System</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface UserMenuProps {
  readonly user?: { readonly email?: string | null; readonly displayName?: string | null };
  readonly onSignOut: () => void;
}

function UserMenu({ user, onSignOut }: UserMenuProps) {
  if (!user) {
    return <Skeleton className="h-9 w-full" />;
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="group/menu-item relative flex w-full items-center gap-2 overflow-hidden rounded-md p-2 text-left text-sm outline-hidden transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 focus-visible:ring-sidebar-ring"
        >
          <UserAvatar email={user.email} name={user.displayName} className="size-7" />
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="truncate font-medium">
              {user.displayName ?? user.email ?? "You"}
            </span>
            {user.displayName && user.email && (
              <span className="text-muted-foreground truncate text-xs">
                {user.email}
              </span>
            )}
          </div>
          <Key className="size-4 shrink-0 opacity-50" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" side="top" className="w-56">
        <DropdownMenuLabel>{user.email ?? "Signed in"}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => (window.location.href = "/settings")}>
          <Gear className="size-4" />
          Settings
        </DropdownMenuItem>
        <DropdownMenuItem onClick={onSignOut}>
          <LogOut className="size-4" />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}