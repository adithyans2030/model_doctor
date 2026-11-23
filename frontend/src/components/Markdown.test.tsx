import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Markdown } from "./Markdown";

afterEach(cleanup);

describe("Markdown", () => {
  it("renders bold, code, headings and both list types", () => {
    const { container } = render(
      <Markdown text={"## Summary\nFix **leakage** first in `cancellation_request`.\n\n* one\n* two\n1. alpha\n2. beta"} />,
    );
    expect(container.querySelector("strong")?.textContent).toBe("leakage");
    expect(container.querySelector("code")?.textContent).toBe("cancellation_request");
    expect(container.querySelectorAll("ul li")).toHaveLength(2);
    expect(container.querySelectorAll("ol li")).toHaveLength(2);
    expect(screen.getByText("Summary").className).toContain("font-semibold");
    expect(container.textContent).not.toContain("**");
  });

  it("never turns model output into markup", () => {
    const { container } = render(<Markdown text={'<img src=x onerror="alert(1)"> **<b>bold</b>**'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect(container.textContent).toContain('<img src=x onerror="alert(1)">');
  });
});
