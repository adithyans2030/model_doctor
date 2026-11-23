import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

/** Serious or critical WCAG A/AA violations on the current page. */
async function axeViolations(page: Page) {
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  return results.violations
    .filter((v) => v.impact === "serious" || v.impact === "critical")
    .map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(", ")}`);
}

function watch(page: Page) {
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`console: ${m.text()}`); });
  return problems;
}

test.describe("landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
  });

  test("renders every section, links to the demo, and has no a11y violations", async ({ page }) => {
    const problems = watch(page);
    await page.goto("/");
    await expect(page).toHaveTitle(/ModelDoctor/);
    await expect(page.getByRole("heading", { level: 1 })).toContainText("Find out why your model underperforms");
    for (const name of ["Models fail quietly.", "From a CSV to an approved model, in five steps.", "Deterministic diagnosis. AI explanation only.",
      "What it looks like on the demo dataset.", "Where it fits.", "Run it locally in one command."]) {
      await expect(page.getByRole("heading", { level: 2, name })).toBeVisible();
    }
    await expect(page.getByText("Built by Adithyan Sabu.")).toBeVisible();
    await expect(page.getByText("synthetic demo dataset, held-out test set")).toBeVisible();
    expect(await axeViolations(page)).toEqual([]);

    // The demo walkthrough tabs switch screenshots.
    await page.getByRole("tab", { name: "Approval" }).click();
    await expect(page.getByRole("img", { name: /Approval gate/ })).toBeVisible();

    await page.getByRole("link", { name: "Try the demo" }).first().click();
    await expect(page).toHaveURL(/\/app$/);
    expect(problems).toEqual([]);
  });

  test("makes no API calls", async ({ page }) => {
    const calls: string[] = [];
    page.on("request", (r) => { if (r.url().includes("/api/")) calls.push(r.url()); });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    expect(calls).toEqual([]);
  });

  test("copy button, theme toggle and mobile menu work", async ({ page, context, browserName }) => {
    if (browserName === "chromium") await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.goto("/");
    await page.getByRole("button", { name: "Copy terminal to clipboard" }).click();
    await expect(page.getByText("Copied to clipboard")).toBeAttached();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain("docker compose up --build");

    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light"); // persisted, applied before paint
    expect(await axeViolations(page)).toEqual([]);

    await page.setViewportSize({ width: 375, height: 800 });
    await page.getByRole("button", { name: "Open menu" }).click();
    await expect(page.locator("#mobile-nav").getByRole("link", { name: "How it works" })).toBeVisible();
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollWidth).toBeLessThanOrEqual(375);
  });
});

test.describe("dashboard happy path", () => {
  test("load the demo dataset, see the leak, read the diagnosis", async ({ page }) => {
    const problems = watch(page);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/app");
    await page.getByRole("button", { name: "Load demo dataset" }).click();
    await expect(page).toHaveURL(/\/app\/datasets\/[^/]+\?tab=data$/, { timeout: 60_000 });
    await expect(page.getByRole("button", { name: /Possible target leakage: 'cancellation_request'/ })).toBeVisible({ timeout: 60_000 });
    expect(await axeViolations(page)).toEqual([]);

    // Tabs are keyboard operable: arrow keys move between them.
    await page.getByRole("tab", { name: "Data Doctor" }).focus();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("heading", { name: "Missing values by column" })).toBeVisible();
    await expect(page.getByRole("img", { name: "Class balance" })).toBeVisible();
    expect(await axeViolations(page)).toEqual([]);

    // Before any model exists, the diagnosis still reports data findings, ranked by severity.
    await page.getByRole("tab", { name: "Diagnosis" }).click();
    await expect(page.getByRole("heading", { name: "Issues, ranked by severity" })).toBeVisible({ timeout: 60_000 });
    await expect(page.getByRole("region", { name: "AI explanation" })).toContainText("explains deterministic results");
    expect(await axeViolations(page)).toEqual([]);
    expect(problems).toEqual([]);
  });
});
