import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  Badge, Button, Card, CodeBlock, ConfirmDialog, DeltaBadge, EmptyState, ErrorState, LoadingRegion, MetricTile, Modal,
  Skeleton, Table, TabPanel, Tabs, TBody, TD, TH, THead, TR, ToastProvider, Tooltip, deltaSentiment, useToast,
} from ".";

describe("Button", () => {
  it("defaults to type=button and fires onClick", async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Save</Button>);
    const btn = screen.getByRole("button", { name: "Save" });
    expect(btn.getAttribute("type")).toBe("button");
    await userEvent.click(btn);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("is disabled and busy while loading", async () => {
    const onClick = vi.fn();
    render(<Button loading onClick={onClick}>Train</Button>);
    const btn = screen.getByRole("button", { name: "Train" });
    expect(btn).toHaveProperty("disabled", true);
    expect(btn.getAttribute("aria-busy")).toBe("true");
    await userEvent.click(btn);
    expect(onClick).not.toHaveBeenCalled();
  });
});

describe("Badge", () => {
  it("renders its label and tone classes", () => {
    render(<Badge tone="danger">Critical</Badge>);
    expect(screen.getByText("Critical").className).toContain("text-danger");
  });
});

describe("Card", () => {
  it("is a section labelled by its title", () => {
    render(<Card title="Leaderboard" actions={<Button>Refresh</Button>}>rows</Card>);
    expect(screen.getByRole("region", { name: "Leaderboard" })).toBeTruthy();
    expect(screen.getByRole("heading", { level: 2, name: "Leaderboard" })).toBeTruthy();
  });
});

function TabsHarness() {
  const [tab, setTab] = useState<"a" | "b" | "c">("a");
  return (
    <>
      <Tabs label="Views" idBase="t" value={tab} onChange={setTab}
        items={[{ value: "a", label: "Alpha" }, { value: "b", label: "Beta", count: 3 }, { value: "c", label: "Gamma" }]} />
      <TabPanel idBase="t" value={tab}>panel {tab}</TabPanel>
    </>
  );
}

describe("Tabs", () => {
  it("uses roving tabindex and links tabs to panels", () => {
    render(<TabsHarness />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.tabIndex)).toEqual([0, -1, -1]);
    const panel = screen.getByRole("tabpanel");
    expect(panel.getAttribute("aria-labelledby")).toBe(tabs[0].id);
    expect(tabs[0].getAttribute("aria-controls")).toBe(panel.id);
  });

  it("moves selection with arrow keys, Home and End", async () => {
    render(<TabsHarness />);
    const user = userEvent.setup();
    screen.getAllByRole("tab")[0].focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { selected: true }).textContent).toContain("Beta");
    expect(document.activeElement?.textContent).toContain("Beta");
    await user.keyboard("{End}");
    expect(screen.getByRole("tabpanel").textContent).toBe("panel c");
    await user.keyboard("{ArrowRight}"); // wraps
    expect(screen.getByRole("tab", { selected: true }).textContent).toBe("Alpha");
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { selected: true }).textContent).toBe("Gamma");
    await user.keyboard("{Home}");
    expect(screen.getByRole("tab", { selected: true }).textContent).toBe("Alpha");
  });
});

describe("Table", () => {
  it("is a focusable labelled scroll region with a caption", () => {
    render(
      <Table label="Per-class metrics">
        <THead><TR><TH>Class</TH><TH numeric>Recall</TH></TR></THead>
        <TBody><TR><TD>churn</TD><TD numeric>0.49</TD></TR></TBody>
      </Table>,
    );
    const region = screen.getByRole("region", { name: "Per-class metrics" });
    expect(region.tabIndex).toBe(0);
    expect(screen.getByRole("table", { name: "Per-class metrics" })).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "Recall" }).className).toContain("text-right");
  });
});

describe("Modal", () => {
  it("renders an accessible dialog and closes on Escape and the close button", () => {
    const onClose = vi.fn();
    const { rerender } = render(<Modal open onClose={onClose} title="Approve v6" description="Moves to approved.">body</Modal>);
    const dialog = screen.getByRole("dialog", { hidden: true });
    expect(dialog.getAttribute("aria-labelledby")).toBeTruthy();
    expect(screen.getByText("Approve v6")).toBeTruthy();
    fireEvent(dialog, new Event("cancel", { cancelable: true })); // what Escape dispatches
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Close dialog", hidden: true }));
    expect(onClose).toHaveBeenCalledTimes(2);
    rerender(<Modal open={false} onClose={onClose} title="Approve v6">body</Modal>);
    expect(screen.queryByText("body")).toBeNull();
  });

  it("ConfirmDialog calls confirm and cancel", async () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(<ConfirmDialog open title="Delete dataset?" confirmLabel="Delete" onConfirm={onConfirm} onCancel={onCancel} />);
    await userEvent.click(screen.getByRole("button", { name: "Delete", hidden: true }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel", hidden: true }));
    expect(onConfirm).toHaveBeenCalledOnce();
    expect(onCancel).toHaveBeenCalledOnce();
  });
});

describe("Tooltip", () => {
  it("shows on focus, describes the trigger and hides on Escape", async () => {
    render(<Tooltip content="Only admins can approve"><button type="button">Approve</button></Tooltip>);
    const trigger = screen.getByRole("button", { name: "Approve" });
    const tip = screen.getByRole("tooltip", { hidden: true });
    expect(trigger.getAttribute("aria-describedby")).toBe(tip.id);
    expect(tip.className).toContain("invisible");
    act(() => trigger.focus());
    expect(tip.className).not.toContain("invisible");
    await userEvent.keyboard("{Escape}");
    expect(tip.className).toContain("invisible");
  });
});

describe("Skeleton / LoadingRegion", () => {
  it("announces loading while skeletons stay hidden from assistive tech", () => {
    render(<LoadingRegion label="Loading models"><Skeleton className="h-4" /></LoadingRegion>);
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("Loading models");
    expect(status.getAttribute("aria-busy")).toBe("true");
    expect(status.querySelector("[aria-hidden]")).toBeTruthy();
  });
});

describe("EmptyState / ErrorState", () => {
  it("renders title, description and action", () => {
    render(<EmptyState title="No models yet" description="Train baselines first." action={<Button>Train</Button>} />);
    expect(screen.getByText("No models yet")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Train" })).toBeTruthy();
  });

  it("shows the error message and retries", async () => {
    const onRetry = vi.fn();
    render(<ErrorState error={new Error("Can't reach the API")} onRetry={onRetry} />);
    expect(screen.getByRole("alert").textContent).toContain("Can't reach the API");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

function ToastHarness() {
  const toast = useToast();
  return <Button onClick={() => toast({ title: "Model approved", tone: "success", duration: 0 })}>Notify</Button>;
}

describe("Toast", () => {
  it("adds a dismissible notification in a live region", async () => {
    render(<ToastProvider><ToastHarness /></ToastProvider>);
    await userEvent.click(screen.getByRole("button", { name: "Notify" }));
    expect(screen.getByRole("region", { name: "Notifications" }).textContent).toContain("Model approved");
    await userEvent.click(screen.getByRole("button", { name: "Dismiss notification" }));
    expect(screen.queryByText("Model approved")).toBeNull();
  });

  it("throws a clear error outside the provider", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<ToastHarness />)).toThrow("useToast must be used inside <ToastProvider>");
  });
});

describe("CodeBlock", () => {
  it("copies the code and confirms", async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText");
    render(<CodeBlock label="Terminal" code={"# start everything\ndocker compose up"} />);
    await user.click(screen.getByRole("button", { name: "Copy terminal to clipboard" }));
    expect(writeText).toHaveBeenCalledWith("# start everything\ndocker compose up");
    expect(await screen.findByText("Copied to clipboard")).toBeTruthy();
  });
});

describe("MetricTile / deltas", () => {
  it("classifies deltas relative to the better direction", () => {
    expect(deltaSentiment(0.44)).toBe("better");
    expect(deltaSentiment(-0.1)).toBe("worse");
    expect(deltaSentiment(-12, false)).toBe("better"); // RMSE going down
    expect(deltaSentiment(0)).toBe("flat");
  });

  it("renders before → after with an accessible delta", () => {
    render(<MetricTile label="Churn recall" before="5%" value="49%"
      delta={{ value: 0.44, format: (v) => `${Math.round(v * 100)} pts`, label: "vs leak-free baseline" }} />);
    expect(screen.getByText("49%")).toBeTruthy();
    expect(screen.getByText("from")).toBeTruthy();
    expect(screen.getByText(/better, vs leak-free baseline/)).toBeTruthy();
  });

  it("DeltaBadge uses a true minus sign for decreases", () => {
    render(<DeltaBadge delta={{ value: -0.05, higherIsBetter: true }} />);
    expect(screen.getByText(/−0.050/)).toBeTruthy();
  });
});
