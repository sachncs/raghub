import * as React from "react";

import { Hero } from "@/components/marketing/hero";
import { TrustStrip } from "@/components/marketing/trust-strip";
import { FeatureGrid } from "@/components/marketing/feature-grid";
import { PolicySection } from "@/components/marketing/policy-section";
import { PlatformPreview } from "@/components/marketing/platform-preview";
import { FinalCta } from "@/components/marketing/final-cta";

export default function MarketingHome() {
  return (
    <>
      <Hero />
      <TrustStrip />
      <FeatureGrid />
      <PolicySection />
      <PlatformPreview />
      <FinalCta />
    </>
  );
}