import { expect, test, type Page } from "@playwright/test";
import { fileURLToPath } from "node:url";

const SAMPLE = (name: string) => fileURLToPath(new URL(`../../sample_data/${name}`, import.meta.url));

/** Fail the test on any uncaught page error, console error or failed API call. */
function watch(page: Page) {
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`console: ${m.text()}`); });
  page.on("response", (r) => { if (r.url().includes("/api/") && r.status() >= 500) problems.push(`HTTP ${r.status()} ${r.url()}`); });
  return problems;
}

async function newProjectWithDataset(page: Page, project: string, file: string, target: string) {
  await page.goto("/");
  await page.getByPlaceholder("Project name").fill(project);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: project })).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles(SAMPLE(file));
  await page.getByRole("button", { name: "Upload & profile" }).click();
  await expect(page.getByRole("heading", { name: file })).toBeVisible();
  await page.locator("select").first().selectOption(target);
  await page.getByRole("button", { name: "Set target" }).click();
  await expect(page.getByText(`target: ${target}`)).toBeVisible();
}

async function trainAndOpenBest(page: Page, algorithms: string[]) {
  await page.getByRole("tab", { name: "Models" }).click();
  const boxes = page.locator('section:has(h2:text("Train baseline models")) input[type="checkbox"]');
  await expect(boxes.first()).toBeVisible();
  for (const box of await boxes.all()) {
    const label = (await box.locator("..").innerText()).trim();
    const want = algorithms.some((a) => label.startsWith(a));
    if ((await box.isChecked()) !== want && (await box.isEnabled())) await box.click();
  }
  await page.getByRole("combobox").filter({ hasText: "off" }).selectOption("0");
  await page.getByRole("button", { name: "Train", exact: true }).click();
  await expect(page.getByText("completed:")).toBeVisible({ timeout: 180_000 });
  const leaderboard = page.locator('section:has(h2:text("Leaderboard"))');
  await leaderboard.getByRole("link").first().click();
}

test("classification: diagnose, explain, optimise, approve, ask", async ({ page }) => {
  const problems = watch(page);
  await newProjectWithDataset(page, "E2E churn", "churn_demo.csv", "churned");

  await page.getByRole("tab", { name: "Data Doctor" }).click();
  await expect(page.getByText("Possible target leakage: 'cancellation_request'")).toBeVisible({ timeout: 60_000 });

  await trainAndOpenBest(page, ["Logistic Regression"]);
  await expect(page.getByText("MACRO-F1 (primary)")).toBeVisible();
  await expect(page.getByText("Confusion matrix (test set)")).toBeVisible();

  await page.getByRole("tab", { name: "Explainability" }).click();
  await expect(page.getByText("What drives the model")).toBeVisible({ timeout: 60_000 });
  await page.locator("select").first().selectOption({ index: 1 });
  await expect(page.getByText(/^Predicted '/)).toBeVisible();
  await page.getByLabel("Explanation method").selectOption("lime");
  await expect(page.getByText(/fidelity/)).toBeVisible();

  await page.getByRole("tab", { name: "Experiments" }).click();
  await page.locator("select").first().selectOption("quick");
  await page.getByRole("button", { name: "Run experiments" }).click();
  await expect(page.getByText("Latest result")).toBeVisible({ timeout: 180_000 });
  await expect(page.getByText("Baseline (leaking features removed)")).toBeVisible();

  await page.getByRole("tab", { name: "Registry" }).click();
  await page.getByPlaceholder("Note (required").fill("e2e approval");
  const ack = page.getByLabel("I acknowledge these risks");
  if (await ack.count()) await ack.check();
  await page.getByRole("button", { name: "Approve" }).click();
  await expect(page.getByText("candidate → approved")).toBeVisible();

  await page.getByRole("tab", { name: "Assistant" }).click();
  await page.getByRole("button", { name: "What should I fix first?" }).click();
  await expect(page.getByText(/rule-based/)).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('[data-role="assistant"]').last()).toContainText("Remove leaking features");

  expect(problems).toEqual([]);
});

test("regression: target summary, R² leaderboard, predicted-vs-actual chart", async ({ page }) => {
  const problems = watch(page);
  await newProjectWithDataset(page, "E2E houses", "house_prices_demo.csv", "price");
  await expect(page.getByText("Target distribution (regression)")).toBeVisible();
  await expect(page.getByText(/strongly skewed/)).toBeVisible();

  await trainAndOpenBest(page, ["Ridge Regression"]);
  await expect(page.getByText("R² (primary)")).toBeVisible();
  await expect(page.getByRole("img", { name: "Predicted versus actual values" })).toBeVisible();
  await expect(page.getByText("Residuals (actual − predicted)")).toBeVisible();
  await page.screenshot({ path: "test-results/regression-performance.png", fullPage: true });

  await page.getByRole("tab", { name: "Explainability" }).click();
  await expect(page.getByText(/Mean absolute error/)).toBeVisible({ timeout: 60_000 });
  await page.locator("select").first().selectOption({ index: 1 });
  await expect(page.getByText(/^Predicted \d/)).toBeVisible();
  await page.screenshot({ path: "test-results/regression-explain.png", fullPage: true });
  expect(problems).toEqual([]);
});

test("unknown pages show the 404 view", async ({ page }) => {
  await page.goto("/definitely/not/here");
  await expect(page.getByText("Page not found")).toBeVisible();
});
