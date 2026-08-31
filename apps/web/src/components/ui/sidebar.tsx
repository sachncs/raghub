"use client"

import * as React from "react"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const SIDEBAR_WIDTH = "15rem"
const SIDEBAR_WIDTH_ICON = "3rem"

type SidebarContextProps = {
  state: "expanded" | "collapsed"
  open: boolean
  setOpen: (open: boolean) => void
  toggle: () => void
  isMobile: boolean
}

const SidebarContext = React.createContext<SidebarContextProps | null>(null)

function useSidebar(): SidebarContextProps {
  const ctx = React.useContext(SidebarContext)
  if (!ctx) {
    throw new Error("useSidebar must be used within a SidebarProvider.")
  }
  return ctx
}

interface SidebarProviderProps {
  readonly defaultOpen?: boolean
  readonly open?: boolean
  readonly onOpenChange?: (open: boolean) => void
  readonly className?: string
  readonly style?: React.CSSProperties
  readonly children: React.ReactNode
}

function SidebarProvider({
  defaultOpen = true,
  open: openProp,
  onOpenChange,
  className,
  style,
  children,
}: SidebarProviderProps) {
  const [openInternal, setOpenInternal] = React.useState(defaultOpen)
  const open = openProp ?? openInternal
  const setOpen = React.useCallback(
    (next: boolean) => {
      if (openProp === undefined) setOpenInternal(next)
      onOpenChange?.(next)
    },
    [onOpenChange, openProp]
  )
  const toggle = React.useCallback(() => setOpen(!open), [open, setOpen])

  const ctx: SidebarContextProps = {
    state: open ? "expanded" : "collapsed",
    open,
    setOpen,
    toggle,
    isMobile: false,
  }

  return (
    <SidebarContext.Provider value={ctx}>
      <div
        data-slot="sidebar-provider"
        style={
          {
            "--sidebar-width": SIDEBAR_WIDTH,
            "--sidebar-width-icon": SIDEBAR_WIDTH_ICON,
            ...style,
          } as React.CSSProperties
        }
        className={cn(
          "group/sidebar-wrapper flex min-h-svh w-full has-data-[variant=inset]:bg-sidebar",
          className
        )}
      >
        {children}
      </div>
    </SidebarContext.Provider>
  )
}

type SidebarState = "expanded" | "collapsed"

interface SidebarProps extends React.ComponentProps<"aside"> {
  readonly side?: "left" | "right"
  readonly state?: SidebarState
}

function Sidebar({ side = "left", state, className, ...props }: SidebarProps) {
  const ctx = useSidebar()
  const resolvedState: SidebarState = state ?? ctx.state
  return (
    <aside
      data-slot="sidebar"
      data-side={side}
      data-state={resolvedState}
      className={cn(
        "bg-sidebar text-sidebar-foreground flex h-full flex-col border-r transition-[width] duration-200 ease-out",
        resolvedState === "expanded"
          ? "w-[--sidebar-width]"
          : "w-[--sidebar-width-icon]",
        className
      )}
      {...props}
    />
  )
}

function SidebarHeader({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-header"
      className={cn("flex flex-col gap-2 p-3", className)}
      {...props}
    />
  )
}

function SidebarContent({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-content"
      className={cn(
        "flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-2",
        className
      )}
      {...props}
    />
  )
}

function SidebarFooter({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-footer"
      className={cn("flex flex-col gap-2 p-3", className)}
      {...props}
    />
  )
}

function SidebarGroup({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-group"
      className={cn("relative flex w-full min-w-0 flex-col p-2", className)}
      {...props}
    />
  )
}

function SidebarGroupLabel({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-group-label"
      className={cn(
        "text-muted-foreground ring-sidebar-ring flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium uppercase tracking-wider outline-hidden transition-[margin,opacity] duration-200 focus-visible:ring-2 [&>svg]:size-4 [&>svg]:shrink-0",
        "group-data-[state=collapsed]/sidebar-wrapper:-mt-8 group-data-[state=collapsed]/sidebar-wrapper:opacity-0",
        className
      )}
      {...props}
    />
  )
}

function SidebarGroupContent({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-group-content"
      className={cn("flex w-full min-w-0 flex-col gap-0.5", className)}
      {...props}
    />
  )
}

function SidebarMenu({
  className,
  ...props
}: React.ComponentProps<"ul">) {
  return (
    <ul
      data-slot="sidebar-menu"
      className={cn("flex w-full min-w-0 flex-col gap-0.5", className)}
      {...props}
    />
  )
}

function SidebarMenuItem({
  className,
  ...props
}: React.ComponentProps<"li">) {
  return (
    <li
      data-slot="sidebar-menu-item"
      className={cn("group/menu-item relative", className)}
      {...props}
    />
  )
}

type ButtonSize = "default" | "sm" | "lg"

interface SidebarMenuButtonProps extends React.ComponentProps<"button"> {
  readonly asChild?: boolean
  readonly isActive?: boolean
  readonly tooltip?: string
  readonly size?: ButtonSize
}

function SidebarMenuButton({
  asChild = false,
  isActive = false,
  size = "default",
  tooltip,
  className,
  ...props
}: SidebarMenuButtonProps) {
  const Comp = asChild ? Slot.Root : "button"
  const sizeClass =
    size === "sm"
      ? "h-8 text-xs"
      : size === "lg"
        ? "h-10 text-sm"
        : "h-9 text-sm"
  return (
    <Comp
      data-slot="sidebar-menu-button"
      data-size={size}
      data-active={isActive}
      title={tooltip}
      className={cn(
        "peer/menu-button ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground data-[active=true]:bg-sidebar-accent data-[active=true]:text-sidebar-accent-foreground data-[active=true]:font-medium flex w-full items-center gap-2 overflow-hidden rounded-md p-2 text-left text-sm outline-hidden transition-[width,height,padding] focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
        sizeClass,
        className
      )}
      {...props}
    />
  )
}

function SidebarRail({
  className,
  ...props
}: React.ComponentProps<"button">) {
  const ctx = useSidebar()
  return (
    <button
      data-slot="sidebar-rail"
      type="button"
      aria-label="Toggle sidebar"
      onClick={ctx.toggle}
      className={cn(
        "hover:after:bg-sidebar-border absolute inset-y-0 z-20 hidden w-4 -translate-x-1/2 transition-all ease-linear after:absolute after:inset-y-0 after:left-1/2 after:w-[2px] group-data-[side=left]:-right-4 group-data-[side=right]:left-0 sm:flex",
        className
      )}
      {...props}
    />
  )
}

function SidebarInset({
  className,
  ...props
}: React.ComponentProps<"main">) {
  return (
    <main
      data-slot="sidebar-inset"
      className={cn(
        "bg-background relative flex w-full flex-1 flex-col",
        "md:peer-data-[variant=inset]:m-2 md:peer-data-[state=collapsed]:peer-data-[variant=inset]:ml-2 md:peer-data-[variant=inset]:ml-0 md:peer-data-[variant=inset]:rounded-xl md:peer-data-[variant=inset]:shadow-sm",
        className
      )}
      {...props}
    />
  )
}

export {
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
  useSidebar,
}