import * as React from "react";

import { Header } from "@/components/marketing/header";
import { Footer } from "@/components/marketing/footer";

interface MarketingShellProps {
  readonly children: React.ReactNode;
  readonly hideHeader?: boolean;
}

export function MarketingShell({ children, hideHeader }: MarketingShellProps) {
  return (
    <div className="relative isolate min-h-svh overflow-hidden">
      <div
        aria-hidden
        className="absolute inset-0 -z-10 gradient-mesh dark:gradient-mesh-dark"
      />
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 -z-10 h-[60vh] bg-gradient-to-b from-background/40 to-transparent"
      />
      {!hideHeader && <Header />}
      <main className="flex flex-1 flex-col">{children}</main>
      <Footer />
    </div>
  );
}