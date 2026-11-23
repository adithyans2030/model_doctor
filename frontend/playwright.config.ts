import { defineConfig, devices } from "@playwright/test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// The e2e run starts its own backend (fresh temporary database, rule-based assistant so
// answers are deterministic) and its own Vite dev server, on ports that don't clash with
// a locally running app.
const API_PORT = 8799;
const WEB_PORT = 4179;
const dataDir = mkdtempSync(join(tmpdir(), "modeldoctor-e2e-"));
const python = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");

export default defineConfig({
  testDir: "e2e",
  timeout: 240_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{
    name: "chromium",
    // Locally use the installed Chrome; CI installs Playwright's Chromium.
    use: { ...devices["Desktop Chrome"], ...(process.env.CI ? {} : { channel: "chrome" }) },
  }],
  webServer: [
    {
      command: `${python} -m alembic upgrade head && ${python} -m uvicorn app.main:create_app --factory --port ${API_PORT}`,
      cwd: "../backend",
      url: `http://127.0.0.1:${API_PORT}/api/v1/health`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        MD_DATA_DIR: dataDir,
        MD_ASSISTANT_MODE: "rules",
        MD_RATE_LIMIT_PER_MINUTE: "0",
        MD_CORS_ORIGINS: `["http://localhost:${WEB_PORT}"]`,
      },
    },
    {
      command: `npx vite --port ${WEB_PORT} --strictPort`,
      url: `http://localhost:${WEB_PORT}`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: { VITE_API_URL: `http://localhost:${API_PORT}` },
    },
  ],
});
