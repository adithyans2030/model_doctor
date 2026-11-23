import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";

afterEach(cleanup);

let shouldThrow = true;
function Flaky() {
  if (shouldThrow) throw new Error("boom");
  return <p>recovered</p>;
}

describe("ErrorBoundary", () => {
  it("contains a crash and can retry", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(<div><p>sibling</p><ErrorBoundary label="the assistant"><Flaky /></ErrorBoundary></div>);
    expect(screen.getByRole("alert").textContent).toContain("Something went wrong in the assistant");
    expect(screen.getByText("boom")).toBeTruthy();
    expect(screen.getByText("sibling")).toBeTruthy(); // the rest of the page survives
    shouldThrow = false;
    fireEvent.click(screen.getByText("Try again"));
    expect(screen.getByText("recovered")).toBeTruthy();
  });
});
