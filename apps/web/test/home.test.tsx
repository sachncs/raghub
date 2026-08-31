import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import Home from "@/app/(marketing)/page";
import { MarketingShell } from "@/components/marketing-shell";

function renderHome() {
  return render(
    <MarketingShell>
      <Home />
    </MarketingShell>
  );
}

describe("Home page", () => {
  it("renders the hero headline and trust micro-line", () => {
    renderHome();
    expect(
      screen.getByRole("heading", { name: /governed hybrid retrieval/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/soc 2 type ii/i)).toBeInTheDocument();
  });

  it("renders the trust, capabilities, policy, and platform sections", () => {
    renderHome();
    expect(
      screen.getByRole("heading", {
        name: /teams who treat retrieval like infrastructure/i,
      })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /every retrieval, every source/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: /document-level access control/i,
      })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: /retrieval layer your team will actually own/i,
      })
    ).toBeInTheDocument();
  });

  it("renders the final CTA with all three call-to-action targets", () => {
    renderHome();
    expect(
      screen.getByRole("heading", { name: /retrieval you can run/i })
    ).toBeInTheDocument();
    const workspaceLinks = screen.getAllByRole("link", {
      name: /create workspace/i,
    });
    expect(workspaceLinks.length).toBeGreaterThanOrEqual(2);
    const selfHostLinks = screen.getAllByRole("link", { name: /self-host/i });
    expect(selfHostLinks.length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getByRole("link", { name: /platform team/i })
    ).toBeInTheDocument();
  });

  it("exposes footer destinations for documentation, security, and pricing", () => {
    renderHome();
    expect(
      screen.getAllByRole("link", { name: /documentation/i }).length
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByRole("link", { name: /security/i }).length
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByRole("link", { name: /pricing/i }).length
    ).toBeGreaterThanOrEqual(1);
  });
});