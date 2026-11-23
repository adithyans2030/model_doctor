import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";
import type { Finding } from "@/lib/types";
import { FindingCard, FindingsList } from "./FindingCard";
import { QueryView } from "./QueryView";
import { findingAnchor, sortBySeverity } from "./severity";

const finding = (over: Partial<Finding>): Finding => ({
  check: "data.missing_values", category: "data", severity: "info", title: "Missing values", evidence: {}, cause: null,
  impact: null, recommendation: null, interventions: [], columns: [], ...over,
});

describe("sortBySeverity", () => {
  it("puts critical first and keeps API order within a severity", () => {
    const sorted = sortBySeverity([
      finding({ title: "i1" }), finding({ title: "w1", severity: "warning" }), finding({ title: "c1", severity: "critical" }),
      finding({ title: "w2", severity: "warning" }),
    ]);
    expect(sorted.map((f) => f.title)).toEqual(["c1", "w1", "w2", "i1"]);
  });

  it("builds stable anchor ids from check ids", () => {
    expect(findingAnchor("feature.leakage")).toBe("finding-feature-leakage");
  });
});

describe("FindingCard", () => {
  it("shows evidence, explanation and the recommended fix; critical findings start open", async () => {
    render(<FindingCard finding={finding({
      severity: "critical", check: "feature.leakage", title: "Possible target leakage",
      evidence: { roc_auc: 0.95112, column: "cancellation_request" }, cause: "Recorded after the outcome.",
      recommendation: "Remove it.", interventions: ["drop_leaky_features"],
    })} />);
    const toggle = screen.getByRole("button", { name: /Possible target leakage/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("0.9511")).toBeTruthy();
    expect(screen.getByText("Remove it.")).toBeTruthy();
    expect(screen.getByText("drop leaky features")).toBeTruthy();
    await userEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
  });
});

describe("FindingsList", () => {
  it("ranks by severity and filters with pressed-state buttons", async () => {
    render(<FindingsList findings={[finding({ title: "Info one" }), finding({ title: "Critical one", severity: "critical" })]} />);
    const titles = screen.getAllByRole("button", { name: /(Info|Critical) one/ }).map((b) => b.textContent);
    expect(titles[0]).toContain("Critical one");
    const filters = within(screen.getByRole("group", { name: "Filter findings by severity" }));
    await userEvent.click(filters.getByRole("button", { name: /^Info/ }));
    expect(filters.getByRole("button", { name: /^Info/ }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByText("Critical one")).toBeNull();
  });

  it("renders an empty state", () => {
    render(<FindingsList findings={[]} empty="No data problems found." />);
    expect(screen.getByText("No data problems found.")).toBeTruthy();
  });
});

function withClient(ui: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

function Harness({ fn }: { fn: () => Promise<string[]> }) {
  const q = useQuery({ queryKey: ["t", fn], queryFn: fn });
  return (
    <QueryView query={q} label="Loading models" skeleton={<div>skeleton</div>} isEmpty={(d) => d.length === 0} empty={<p>No models yet</p>}>
      {(d) => <p>{d.join(",")}</p>}
    </QueryView>
  );
}

describe("QueryView", () => {
  it("shows a labelled loading state, then data", async () => {
    render(withClient(<Harness fn={async () => ["a", "b"]} />));
    expect(screen.getByRole("status").textContent).toContain("Loading models");
    expect(await screen.findByText("a,b")).toBeTruthy();
  });

  it("shows the empty state", async () => {
    render(withClient(<Harness fn={async () => []} />));
    expect(await screen.findByText("No models yet")).toBeTruthy();
  });

  it("shows the error with a retry", async () => {
    render(withClient(<Harness fn={async () => { throw new Error("Can't reach the API"); }} />));
    expect((await screen.findByRole("alert")).textContent).toContain("Can't reach the API");
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
  });
});
