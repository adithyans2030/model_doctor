import { useEffect } from "react";
import { Hero } from "./Hero";
import { LandingNav } from "./LandingNav";
import { MotionProvider } from "./Reveal";
import {
  Comparison, DemoWalkthrough, Features, FinalCta, Footer, HowItWorks, Problem, ProofStrip, QuickStart, TechStack, Trust,
} from "./sections";

export default function Landing() {
  useEffect(() => {
    document.title = "ModelDoctor: find out why your ML model underperforms";
  }, []);
  return (
    <MotionProvider>
      <a href="#content" className="sr-only-focusable fixed top-2 left-2 z-[70] rounded-md bg-accent px-3 py-2 text-sm text-accent-ink">Skip to content</a>
      <LandingNav />
      <main id="content">
        <Hero />
        <ProofStrip />
        <Problem />
        <HowItWorks />
        <Features />
        <Trust />
        <DemoWalkthrough />
        <Comparison />
        <TechStack />
        <QuickStart />
        <FinalCta />
      </main>
      <Footer />
    </MotionProvider>
  );
}
